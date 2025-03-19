# References:
#   Mamba:  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
#   DiT:    https://github.com/facebookresearch/DiT/blob/main/models.py#67
#   VAR:    https://github.com/FoundationVision/VAR/blob/main/models/var.py

import torch
import torch.nn as nn
import math
import json
import os
import copy

from collections import namedtuple
from functools import partial
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from .block import Block
from .lora import _find_and_replace
from .generation import GenerationMixin
from transformers.integrations import is_deepspeed_zero3_enabled
from accelerate.hooks import add_hook_to_module
from typing import Optional

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from models.cobra.nn_utils import FusedMLPProjector


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        token_drop=0.0,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
            the project up to embed_dim
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = FusedMLPProjector(word_embed_proj_dim, embed_dim)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )
        self.token_dropout = nn.Dropout(token_drop)
        # nn.init.normal_(self.word_embeddings.weight, std=0.02)

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.token_dropout(self.word_embeddings(input_ids))
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)

            embeddings = embeddings + position_embeddings
        return embeddings


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        embeddings = self.cap_proj(caption)
        return embeddings



def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    adaln_group=False,
    adaln=False,
    n_layer=0,
    mixer_drop=0.0,
    mlp_drop=0.0,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        adaln_group=adaln_group,
        adaln=adaln,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        vqvae_vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        num_tokens=256,
        adaln_group=False,
        token_drop=0.0,
        mixer_drop=0.0,
        mlp_drop=0.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.t2i_task = config.t2i_task
        self.mmu_task = config.mmu_task
        self.stage = 'align'
        if self.t2i_task:
            self.img_embeddings = GPT2Embeddings(d_model, vqvae_vocab_size, -1, token_drop=token_drop, word_embed_proj_dim=d_model, **factory_kwargs)
            self.pos_embed = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, num_tokens + 73, d_model)), 0., 0.02)
            self.caption_embed = CaptionEmbedder(in_channels=d_model, hidden_size=d_model)
        if self.mmu_task:
            self.mmu_pos_embed = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, 1500, d_model)), 0., 0.02)
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.img_sq_len = 729 # siglip+dinov2

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    adaln_group=adaln_group,
                    mixer_drop=mixer_drop,
                    mlp_drop=mlp_drop,
                    adaln=False,
                    n_layer=n_layer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )
        self.layers = _find_and_replace(self.layers)


    def get_input_embeddings(self):
        return self.embedding

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def set_lora_mode(self,task='t2i'):
        if task=='t2i':
            for layer in self.layers:
                layer.mixer.in_proj.task_types = 't2i'
        elif task=='mmu':
            for layer in self.layers:
                layer.mixer.in_proj.task_types = 'mmu'


    # when cond is not None, it is the class label of t2i task
    def forward(self, input_ids, input_embeddings, position_ids, cond, task, inference_params=None, **mixer_kwargs):
        is_train = inference_params is None
        self.set_lora_mode(task)
        if input_embeddings is not None: # for train
            hidden_states = input_embeddings
            if task == 't2i' and cond is not None:
                cond = cond.unsqueeze(1) + self.timesteps_embeddings[:, :hidden_states.shape[1]]
            if task == 'mmu':
                if position_ids is None:
                    position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long, device=hidden_states.device)
                    position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)
                    hidden_states = hidden_states + self.mmu_pos_embed[:, :hidden_states.shape[1]]
        else: # for infer
            if task == 't2i':
                hidden_states = self.img_embeddings(input_ids)
                # add time embedding
                if cond is not None:
                    timesteps_embeddings = self.timesteps_embeddings.repeat(hidden_states.shape[0], 1, 1)
                    cond = cond.unsqueeze(1) + timesteps_embeddings.gather(1, position_ids.unsqueeze(-1).expand(-1, -1, timesteps_embeddings.size(-1)))
                # position embedding
                pos_embed = self.pos_embed.repeat(hidden_states.shape[0], 1, 1)
                pos_embed = pos_embed.gather(1, position_ids.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1)))
                hidden_states = hidden_states + pos_embed
            if task == 'mmu':
                hidden_states = self.embedding(input_ids)
                mmu_pos_embed = self.mmu_pos_embed.repeat(hidden_states.shape[0], 1, 1)
                mmu_pos_embed = mmu_pos_embed.gather(1, position_ids.unsqueeze(-1).expand(-1, -1, mmu_pos_embed.size(-1)))
                hidden_states = hidden_states + mmu_pos_embed

        residual = None
        if task == 't2i' and cond is not None:                
            ada_cond = self.adaln_group(cond).chunk(self.num_groups, dim=2)
            for i, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    hidden_states, residual, ada_cond[i % self.num_groups], task=task, inference_params=inference_params
                )
        elif task == 't2i' and cond is None:
            for i, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    hidden_states, residual, None, task=task, inference_params=inference_params
                )
        elif task == 'mmu':
            for i, layer in enumerate(self.layers): 
                hidden_states, residual = layer(
                    hidden_states, residual, None, task=task, inference_params=inference_params
                )


        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        if task == 't2i' and cond is not None:
            hidden_states = self.final_layer(hidden_states, cond)
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.t2i_task = config.t2i_task
        self.mmu_task = config.mmu_task

        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        vocab_size = config.vocab_size
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            config=config,
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=vocab_size,
            vqvae_vocab_size=config.vqvae_vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            num_tokens=config.num_tokens,
            adaln_group=config.adaln_group,
            token_drop=config.token_drop,
            mixer_drop=config.mixer_drop,
            mlp_drop=config.mlp_drop,
            **factory_kwargs,
        )
        if self.t2i_task:
            self.img_head = nn.Linear(config.d_model, config.vqvae_vocab_size, bias=False, **factory_kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            if self.t2i_task:
                self.img_head.weight = self.backbone.img_embeddings.word_embeddings.weight
            self.lm_head.weight = self.backbone.embedding.weight


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, input_embeddings, position_ids=None, cond=None, task=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        hidden_states = self.backbone(input_ids, input_embeddings, position_ids, cond, task, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        t2i_logits = None
        mmu_logits = None
        if task == 't2i':
            t2i_logits = self.img_head(hidden_states)
        if task == 'mmu':
            mmu_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["t2i_logits", "mmu_logits"])
        return CausalLMOutput(t2i_logits=t2i_logits, mmu_logits=mmu_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()
    def get_output_embeddings(self):
        return self.lm_head
    def set_input_embeddings(self, value):
        self.backbone.embedding = value


    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            print(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        _init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings