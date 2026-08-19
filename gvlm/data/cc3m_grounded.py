"""Dataset and mapper for CC3M with region-level grounding.

`CC3MGroundedWebDataset` reads the packaged webdataset, whose samples are `.jpg`
(shortest side 256), `.txt` (caption) and `.json` (boxes and phrases).

Box crops are cut at load time rather than stored as `parent{i}.jpg` like
utils/prepare_GRIT_webdataset.py does. Dedup, area cutoff and square padding
follow that script.
"""

from __future__ import annotations

import copy
import glob
import random
from typing import Callable

import webdataset as wds
from loguru import logger
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms as T

import gvlm.utils.distributed as dist


def pad_to_square(img, fill=(255, 255, 255)):
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class CC3MGroundedTarMapper:
    """Pre-process one CC3M grounded sample into image/text/box_image/box_text."""

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        min_box_area: int = 32 * 32,
        pad_boxes_to_square: bool = True,
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
            min_box_area: Drop boxes whose crop falls below this area. Applied to
                the stored image, which is downscaled, so unlike the GRIT cutoff
                on full-resolution pixels it bounds what the encoder actually sees.
            pad_boxes_to_square: Pad crops to square so the resize does not stretch them.
        """
        self.image_transform = T.Compose(image_transform)
        self.min_box_area = min_box_area
        self.pad_boxes_to_square = pad_boxes_to_square

    def __call__(self, dataset_dict: dict):
        meta = dataset_dict["json"]
        image = dataset_dict["jpg"]
        iw, ih = image.size

        # Repeated phrases ground onto several boxes, which would read as false
        # negatives in the contrastive loss; keep the first box per phrase.
        seen, candidates = set(), []
        for (px, py, pw, ph), phrase in zip(meta["boxes_xywh_norm"], meta["phrases"]):
            phrase = phrase.strip()
            if not phrase or phrase in seen:
                continue
            box = (
                max(0, int(px * iw)),
                max(0, int(py * ih)),
                min(iw, int((px + pw) * iw)),
                min(ih, int((py + ph) * ih)),
            )
            if (box[2] - box[0]) * (box[3] - box[1]) <= self.min_box_area:
                continue
            seen.add(phrase)
            candidates.append((box, phrase))

        if not candidates:
            return None

        box, box_text = random.choice(candidates)
        crop = image.crop(box)
        if self.pad_boxes_to_square:
            crop = pad_to_square(crop)

        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(image),
            "text": meta["caption"],
            "box_image": self.image_transform(crop),
            "box_text": box_text,
        }


class CC3MGroundedWebDataset(IterableDataset):
    """As ImageTextWebDataset, but drops samples where no box survives the mapper."""

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper, handler=wds.warn_and_continue))
        pipeline.append(wds.select(lambda sample: sample is not None))

        if self.infinite_stream:
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            yield from pipeline
