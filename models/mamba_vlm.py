import torch
import torch.nn as nn
from .stage2.config_mamba import MambaConfig
from .stage2.mixer_seq_simple import MambaLMHeadModel
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer
from typing import Type
from models.cobra.backbones.llm.prompting import (
    PromptBuilder,
    MambaPromptBuilder
)
from models.cobra.prompting_utils import UniversalPrompting
from llamagen_tokenizer.tokenizer_image.vq_model import VQ_models as llamagen_VQ_models

class MambaVLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: MambaConfig):
        super().__init__()
        # init all models
        self.vqvae = self.init_1st_stage_model()
        self.mamba = self.init_2nd_stage_model(config)
        self.identifier = 'MambaVLM'
        self.d_model = config.d_model
        self.num_classes = config.num_classes
        self.num_tokens = config.num_tokens
        self.pad_vocab_size_multiple = config.pad_vocab_size_multiple

        # self.tokenizer = AutoTokenizer.from_pretrained('ckpts/gpt-neox-20b/', model_max_length=2048)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-neox-20b',
            model_max_length=2048
        )
        self.uni_prompting = UniversalPrompting(self.tokenizer, max_text_len=499,
                                        special_tokens=(
                                            "<|soi|>", "<|eoi|>", "<|sot|>", "<|eot|>", "<|t2i|>",
                                            "<|mmu|>", "<|soc|>", "<|eoc|>", "<|lvg|>"
                                        ),
                                        ignore_id=-100, cond_dropout_prob=0.1)

        print('special tokens : \n', self.uni_prompting.sptids_dict)
  

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def get_tokenizer(self):
        return self.tokenizer

    def get_uni_prompting(self):
        return self.uni_prompting

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return MambaPromptBuilder

    #llamagen_t2i
    def init_1st_stage_model(self):
        vq_model = llamagen_VQ_models['VQ-16']()
        # Download at https://huggingface.co/peizesun/llamagen_t2i
        # checkpoint = torch.load('ckpts/llamagen/vq_ds16_t2i.pt', map_location="cpu")
        checkpoint_path = hf_hub_download(
            repo_id="peizesun/llamagen_t2i",
            filename="vq_ds16_t2i.pt",
            revision="main"  # optional: specify branch/tag
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        vq_model.eval()
        [p.requires_grad_(False) for p in vq_model.parameters()]
        return vq_model


    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of):
        self.mamba.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.mamba.get_input_embeddings()(input_ids)

    def init_2nd_stage_model(self, config):
        model = MambaLMHeadModel(config)
        return model

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.mamba.backbone.layers.parameters())
        if non_embedding:
            n_params -= self.mamba.backbone.embeddings.word_embeddings.weight.numel()
        return n_params

    def forward(self, x, c, cond, task='t2i'):
        logits = self.mamba(input_ids=None, input_embeddings=x, cond=cond, task=task)
        target = c
        if task == 't2i':
            logits = logits.t2i_logits
        elif task == 'mmu':
            logits = logits.mmu_logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        # Flatten the tokens
        logits = shift_logits.view(-1, shift_logits.shape[-1])
        target = shift_labels.view(-1)
        
        return logits, target
    
    @torch.no_grad()
    def decode_to_img(self, index):
        z_shape = [index.shape[0], 8, 16, 16]
        x = self.vqvae.decode_code(index, shape=z_shape)
        return x

     

def OmniMamba_L(t2i_task, mmu_task, **kwargs):
    return MambaVLM(MambaConfig(d_model=1024, n_layer=48, adaln_group=True, num_groups=4, t2i_task=t2i_task, mmu_task=mmu_task, **kwargs))

def OmniMamba_1_3B(t2i_task, mmu_task, **kwargs):
    return MambaVLM(MambaConfig(d_model=2048, n_layer=48, adaln_group=True, num_groups=4, t2i_task=t2i_task, mmu_task=mmu_task, **kwargs))



MambaVLMs = {'OmniMamba-L': OmniMamba_L, 'OmniMamba-1.3B': OmniMamba_1_3B}
