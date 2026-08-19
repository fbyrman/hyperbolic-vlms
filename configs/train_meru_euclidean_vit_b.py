from torch.optim.adamw import AdamW

from gvlm.config import LazyCall as L
from gvlm.encoders.image_encoders import build_timm_vit
from gvlm.encoders.text_encoders import TransformerTextEncoder
from gvlm.models import EuclideanMERU
from gvlm.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param

from .cc3m_dataset import dataset  # noqa: F401

model = L(EuclideanMERU)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512",
        vocab_size=49408,
        context_length=77,
    ),
    embed_dim=512,
    use_boxes=False,
    aperture_k=0.2 / 0.1**0.5,
    entail_weight=0.2,
    eps=1e-8,
)


optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=["logit_scale", "visual_alpha", "textual_alpha", "curv"],
        ),
        lr=5e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


train = dict(
    seed=0,
    amp=True,
    total_batch_size=512,
    num_iterations=100000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(broadcast_buffers=False, static_graph=True),
    ddp_fp16_compression=True,
)
