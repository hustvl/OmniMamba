from dataclasses import dataclass, field


@dataclass
class MambaConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    n_layer: int = 48

    # llamagen_t2i
    vqvae_vocab_size: int = 16384
    num_tokens: int = 256

    vocab_size: int = 50277

    ssm_cfg: dict = field(default_factory=lambda: {'layer': 'Mamba2'})
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

    # for mamba_130m
    pad_vocab_size_multiple: int = 16 
    
    tie_embeddings: bool = True
    # update
    num_classes: int = 1000
    
    # adaLN
    adaln_group: bool = False
    num_groups: int = 1
    # dropout
    token_drop: float = 0.0
    mixer_drop: float = 0.0
    mlp_drop: float = 0.0
    # task
    t2i_task: bool = False
    mmu_task: bool = False