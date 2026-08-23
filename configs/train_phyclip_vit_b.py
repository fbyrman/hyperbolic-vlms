# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru
# Modified from github.com/tksmatsubara/PHyCLIP

"""
Each config file should have four dicts or OmegaConf objects:
`dataset`, `model`, `optim`, and `train`.

User can compose config files by importing these objects and overriding specific
parameters. See examples in other training configs.

Reference: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
"""

from torch.optim.adamw import AdamW  # noqa: F401

from gvlm.config import LazyCall as L
from gvlm.encoders.image_encoders import build_timm_vit
from gvlm.encoders.text_encoders import TransformerTextEncoder
from gvlm.models import PHyCLIP
from gvlm.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param

from .cc3m_dataset import dataset  # noqa: F401

model = L(PHyCLIP)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512",
        vocab_size=49408,
        context_length=77,  # originally context_length=77
    ),
    embed_dim=512,
    subspace_dim=8,
    curv_init=1,
    learn_curv=True,
    entail_weight=0.2,
    use_boxes=True,
    lorentz_eps=1e-6,
    product_metric="l1",
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


# Other parameters useful for training script.
train = dict(
    seed=0,
    amp=True,
    total_batch_size=512,
    num_iterations=100000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, static_graph=True
    ),
    ddp_fp16_compression=True,
)
