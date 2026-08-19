"""Shared CC3M grounded training dataset, imported by every method config.

Reads the webdataset built by scripts/extract_gligen_cc3m.py, at
`CC3M_GROUNDED_DIR`.
"""

import os
from pathlib import Path

from torchvision import transforms as T

from gvlm.config import LazyCall as L
from gvlm.data.cc3m_grounded import CC3MGroundedTarMapper, CC3MGroundedWebDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]

_mapper = L(CC3MGroundedTarMapper)(
    image_transform=[
        L(T.RandomResizedCrop)(
            size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
        ),
        L(T.ToTensor)(),
    ],
    min_box_area=32 * 32,
    pad_boxes_to_square=True,
)

_data_dir = os.environ.get(
    "CC3M_GROUNDED_DIR", str(_REPO_ROOT / "datasets" / "cc3m_grounded")
)
dataset = L(CC3MGroundedWebDataset)(
    tarfiles=[os.path.join(_data_dir, "train-*.tar")],
    mapper=_mapper,
    buffer_size=4000,
    seed="${..train.seed}",
)
