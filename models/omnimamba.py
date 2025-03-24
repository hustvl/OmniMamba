"""
cobra.py

PyTorch Module defining a CobraVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""
from __future__ import annotations

from typing import Optional, Type

from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn
from models.cobra.backbones.llm.prompting import PromptBuilder
from models.cobra.overwatch import initialize_overwatch
from models.cobra.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector, FusedLDPProjector
from models.cobra.materialize import get_vision_backbone_and_transform
from models.mamba_vlm import MambaVLMs


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel()}")


class OmniMamba(nn.Module,
                PyTorchModelHubMixin,
                repo_url="https://github.com/hustvl/OmniMamba",
                paper_url="https://arxiv.org/abs/2407.11015",
                pipeline_tag="any-to-any",
                license="mit"):
    def __init__(
        self,
        args,
        arch_specifier= "fused-gelu-mlp",
        stage='finetune',
    ):
        super().__init__()
        self.args = args
        self.vision_backbone, self.image_transform = get_vision_backbone_and_transform(args.image_backbone)
        self.llm_backbone = MambaVLMs[args.omnimamba_model](args.t2i_task, args.mmu_task)
        # Set Weight Initialization Seed for Projector Consistency
        if stage != 'inference':
            torch.manual_seed(self.vision_backbone.embed_dim)
        self.llm_backbone.mamba.backbone.stage = stage
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize Projection (Adapter) based on `arch_specifier`
        if args.mmu_task:
            self.arch_specifier = arch_specifier
            if arch_specifier == "linear":
                self.projector = LinearProjector(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
            elif arch_specifier.endswith("fused-gelu-mlp"):
                self.projector = FusedMLPProjector(self.vision_backbone.embed_dim, self.llm_backbone.d_model)
            elif arch_specifier.endswith("gelu-mlp"):
                self.projector = MLPProjector(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
            elif arch_specifier.endswith("fused-ldpnet"):
                self.projector = FusedLDPProjector(self.vision_backbone.embed_dim, self.llm_backbone.embed_dim)
            else:
                raise ValueError(f"CobraVLM with `{arch_specifier = }` is not supported!")

            
        self.eos_token_id = self.llm_backbone.tokenizer.eos_token_id
        
        self.load_pretrain_model(args)
        overwatch.info("Load pretrain model done", ctx_level=1)
        self.freeze_backbones(stage=stage)
        overwatch.info("Freeze backbones done", ctx_level=1)
        

    def load_pretrain_model(self, args):
        if args.vq_ckpt is not None:
            vqvae_state_dict = torch.load(args.vq_ckpt, map_location="cpu")
            if 'quantize.codebook_used' in vqvae_state_dict:
                vqvae_state_dict.pop('quantize.codebook_used')
            self.llm_backbone.vqvae.load_state_dict(vqvae_state_dict)
        if args.omnimamba_ckpt is not None:
            overwatch.info(f"Loading omnimamba model from {args.omnimamba_ckpt}", ctx_level=1)
            state_dict = torch.load(args.omnimamba_ckpt, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            if args.mamba_pretrain is not None:
                mamba_state_dict = torch.load(args.mamba_pretrain, map_location="cpu")
                self.llm_backbone.mamba.load_state_dict(mamba_state_dict, strict=False) 
        if args.mmu_task:
            self.llm_backbone.resize_token_embeddings(len(self.llm_backbone.uni_prompting.text_tokenizer), pad_to_multiple_of=self.llm_backbone.pad_vocab_size_multiple)         

    def get_projector_weight_dtype(self):
        # 直接获取第一层的 dtype
        for layer in self.projector.projector:
            if isinstance(layer, torch.nn.Linear):
                return layer.weight.dtype

    def allocate_inference_cache(self, *args, **kwargs):
        return self.llm_backbone.allocate_inference_cache(*args, **kwargs)
       

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.train()
            self.vision_backbone.eval()
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.eval()
            self.llm_backbone.requires_grad_(False)
            if self.args.t2i_task:
                self.llm_backbone.mamba.backbone.img_embeddings.train()
                self.llm_backbone.mamba.backbone.img_embeddings.requires_grad_(True)
                self.llm_backbone.mamba.backbone.embedding.requires_grad_(True)
                self.llm_backbone.mamba.backbone.pos_embed.requires_grad_(True)
                self.llm_backbone.mamba.backbone.caption_embed.train()
                self.llm_backbone.mamba.backbone.caption_embed.requires_grad_(True)
                self.llm_backbone.mamba.img_head.train()
                self.llm_backbone.mamba.img_head.requires_grad_(True)
                # lora
                for name, module in self.llm_backbone.mamba.backbone.named_modules():
                    if 'lora' in name.lower():
                        module.train()
                        for param in module.parameters():
                            param.requires_grad_(True)
            if self.args.mmu_task:
                self.projector.train()
                self.projector.requires_grad_(True)
                # lora
                for name, module in self.llm_backbone.mamba.backbone.named_modules():
                    if 'lora' in name.lower():
                        module.train()
                        for param in module.parameters():
                            param.requires_grad_(True)

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            if self.args.mmu_task:
                overwatch.info(f"[TRAINABLE] 🔥 =>> ALL Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.train()
            self.vision_backbone.eval()
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.mamba.train()
            self.llm_backbone.mamba.requires_grad_(True)
            self.llm_backbone.vqvae.eval()
            self.llm_backbone.vqvae.requires_grad_(False)
            if self.args.mmu_task:
                self.projector.train()
                self.projector.requires_grad_(True)

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> VQVAE", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            if self.args.mmu_task:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
        elif stage == 'inference':
            self.eval()
            self.requires_grad_(False)
        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def mmu_multi_ids2embed(self, pixel_values_mmu, input_ids_mmu, labels_mmu):
        input_ids_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|mmu|>']).to(input_ids_mmu.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|soi|>']).to(input_ids_mmu.device),
            # image_tokens_mmu, place img embedding here
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|eoi|>']).to(input_ids_mmu.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|sot|>']).to(input_ids_mmu.device),
            input_ids_mmu,
            # <eot> is in dataset
        ], dim=1).long()

        images_feat = self.vision_backbone(pixel_values_mmu)    
        images_embeddings = self.projector(images_feat)
        text_embeddings = self.llm_backbone.embed_input_ids(input_ids_mmu)

        part1 = text_embeddings[:, :2, :]
        part2 = text_embeddings[:, 2:, :]
        input_embeddings_mmu = torch.cat((part1, images_embeddings, part2), dim=1)

        labels_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # mmu
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # soi
            torch.ones_like(images_embeddings[:, :, 0]) * self.llm_backbone.uni_prompting.ignore_id,  # ignore image embedding
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # eoi
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # sot
            labels_mmu.to(input_ids_mmu.device)
            # <eot> is in dataset
        ], dim=1).long()
        return input_embeddings_mmu, labels_mmu

    # for text only
    def mmu_uni_ids2embed(self, pixel_values_mmu, input_ids_mmu, labels_mmu):
        input_ids_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|mmu|>']).to(input_ids_mmu.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|soi|>']).to(input_ids_mmu.device),
            # image_tokens_mmu, place img embedding here
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|eoi|>']).to(input_ids_mmu.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.sptids_dict['<|sot|>']).to(input_ids_mmu.device),
            input_ids_mmu,
            # <eot> is in dataset
        ], dim=1).long()

        
        text_embeddings = self.llm_backbone.embed_input_ids(input_ids_mmu)
        pad_images_embeddings = torch.zeros(input_ids_mmu.shape[0], self.llm_backbone.mamba.backbone.img_sq_len, text_embeddings.shape[-1], device=text_embeddings.device)        
        
        part1 = text_embeddings[:, :2, :]
        part2 = text_embeddings[:, 2:, :]
        input_embeddings_mmu = torch.cat((part1, pad_images_embeddings, part2), dim=1)
        
        labels_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # mmu
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # soi
            torch.ones_like(pad_images_embeddings[:, :, 0]) * self.llm_backbone.uni_prompting.ignore_id,  # ignore image embedding
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # eoi
            (torch.ones(input_ids_mmu.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(input_ids_mmu.device), # sot
            labels_mmu.to(input_ids_mmu.device)
            # <eot> is in dataset
        ], dim=1).long()
        return input_embeddings_mmu, labels_mmu    


    def forward(self, inputs, task='t2i'):
        if task == 't2i':

            # t2i task
            # pretokenized data
            image_ids_t2i = inputs["t2i_flow"]['inputs']
            caption_ids_t2i = inputs["t2i_flow"]['caption_ids']

            image_embeddings_t2i = self.llm_backbone.mamba.backbone.img_embeddings(image_ids_t2i)

            text_embeddings_t2i = self.llm_backbone.embed_input_ids(caption_ids_t2i)
            text_embeddings_t2i = self.llm_backbone.mamba.backbone.caption_embed(text_embeddings_t2i, train=True)
            input_embeddings_t2i = torch.cat((text_embeddings_t2i[:, :-1, :], image_embeddings_t2i, text_embeddings_t2i[:, -1:, :]), dim=1)


            labels_t2i = torch.cat([
                (torch.ones(image_ids_t2i.shape[0], (caption_ids_t2i.shape[1]-1)) * self.llm_backbone.uni_prompting.ignore_id).to(image_embeddings_t2i.device), # class id
                image_ids_t2i,  # image ids
                (torch.ones(image_ids_t2i.shape[0], 1) * self.llm_backbone.uni_prompting.ignore_id).to(image_embeddings_t2i.device), # eoi
            ], dim=1).long()

            pos_embed = self.llm_backbone.mamba.backbone.pos_embed.repeat(input_embeddings_t2i.shape[0], 1, 1)
            input_embeddings_t2i = input_embeddings_t2i + pos_embed[:, :input_embeddings_t2i.shape[1]]
            logits, target = self.llm_backbone(input_embeddings_t2i, labels_t2i, cond=None, task='t2i')
            logits = logits.view(-1, logits.shape[-1])
            target = target.view(-1)

            loss_t2i = self.loss_fn(logits, target)
            return loss_t2i
        elif task == 'mmu':
            # mmu task
            pixel_values_mmu, input_ids_mmu, labels_mmu, multimodal_indices = (inputs["mmu_flow"]["pixel_values"],
                                                        inputs["mmu_flow"]["input_ids"],
                                                        inputs["mmu_flow"]["labels"],
                                                        inputs["mmu_flow"]["multimodal_indices"])
    
            # no text only data
            if len(multimodal_indices) == input_ids_mmu.shape[0]:
                input_embeddings_mmu, labels_mmu = self.mmu_multi_ids2embed(pixel_values_mmu, input_ids_mmu, labels_mmu)
            # all text only data
            elif len(multimodal_indices) == 0:
                input_embeddings_mmu, labels_mmu = self.mmu_uni_ids2embed(pixel_values_mmu, input_ids_mmu, labels_mmu)
            else:
                all_indices = torch.arange(pixel_values_mmu['dino'].shape[0], device=multimodal_indices.device)
                uni_indices = all_indices[~torch.isin(all_indices, multimodal_indices)]
                # multi
                multi_input_embeddings_mmu, multi_labels_mmu = self.mmu_multi_ids2embed({'dino': pixel_values_mmu['dino'][multimodal_indices], 'siglip': pixel_values_mmu['siglip'][multimodal_indices]}, input_ids_mmu[multimodal_indices], labels_mmu[multimodal_indices])
                # uni text only data
                uni_input_embeddings_mmu, uni_labels_mmu = self.mmu_uni_ids2embed(None, input_ids_mmu[uni_indices], labels_mmu[uni_indices])
                
                input_embeddings_mmu = torch.cat((multi_input_embeddings_mmu, uni_input_embeddings_mmu), dim=0)
                labels_mmu = torch.cat((multi_labels_mmu, uni_labels_mmu), dim=0)

            logits, target = self.llm_backbone(input_embeddings_mmu, labels_mmu, cond=None, task='mmu')
            loss_mmu = self.loss_fn(logits, target)
            return loss_mmu
        


    def t2i_generate(self, text_ids=None, temperature=1.0, top_k=0, top_p=1.0, fast=True):
        caption_embeddings = self.llm_backbone.embed_input_ids(text_ids)
        input_embeddings_t2i = caption_embeddings
        input_embeddings_t2i = self.llm_backbone.mamba.backbone.caption_embed(input_embeddings_t2i, train=False)

        # position embedding
        pos_embed = self.llm_backbone.mamba.backbone.pos_embed.repeat(input_embeddings_t2i.shape[0], 1, 1)
        input_embeddings_t2i = input_embeddings_t2i + pos_embed[:, :input_embeddings_t2i.shape[1]]
        
        input_ids_t2i = text_ids
        max_length = self.llm_backbone.num_tokens + input_embeddings_t2i.shape[1]
        x = self.llm_backbone.mamba.generate(input_ids=input_ids_t2i,
                                input_embeddings=input_embeddings_t2i,
                                cond=None,
                                max_length=max_length,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                cg=fast,
                                task='t2i',)
        
        self.llm_backbone.mamba._decoding_cache = None

        tokens =  x[:text_ids.shape[0], input_embeddings_t2i.shape[1]:]

        imgs = self.llm_backbone.decode_to_img(tokens)
        return imgs
