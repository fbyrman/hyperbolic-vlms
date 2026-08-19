#!/usr/bin/env python
"""Sanity-check extracted shards: decode images, check box/phrase alignment.

Scans a whole shard for structural problems (missing members, box counts that
disagree with phrase counts, boxes outside the image) and writes a few samples
to disk with their boxes drawn, so the grounding can be eyeballed.
"""
import argparse
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = str(REPO_ROOT / "datasets" / "cc3m_grounded")
DEFAULT_SAVE_DIR = str(REPO_ROOT / "outputs" / "sanity_samples")


def check(key, img, meta):
    problems = []
    boxes = meta["boxes_xywh"]
    if not (len(boxes) == len(meta["phrases"]) == len(meta["tokens_positive"])):
        problems.append("boxes/phrases/tokens length mismatch")
    if (meta["width"], meta["height"]) == (0, 0):
        problems.append("zero original size")
    if img.size != (meta["resized_width"], meta["resized_height"]):
        problems.append(f"jpeg {img.size} != meta {(meta['resized_width'], meta['resized_height'])}")
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            problems.append("non-positive box extent")
        if x < -1 or y < -1 or x + w > meta["width"] + 1 or y + h > meta["height"] + 1:
            problems.append("box outside original bounds")
    for nb in meta["boxes_xywh_norm"]:
        if any(v < -0.01 or v > 1.01 for v in nb):
            problems.append("normalized box outside [0,1]")
    if not meta["caption"].strip():
        problems.append("empty caption")
    if any(not p.strip() for p in meta["phrases"]):
        problems.append("empty phrase")
    return problems


def draw_boxes(img, meta):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for (nx, ny, nw, nh), phrase in zip(meta["boxes_xywh_norm"], meta["phrases"]):
        box = [nx * w, ny * h, (nx + nw) * w, (ny + nh) * h]
        d.rectangle(box, outline=(255, 0, 0), width=2)
        d.text((box[0] + 3, box[1] + 3), phrase[:40], fill=(255, 255, 0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", help="shard name, e.g. train-0000; default random")
    args = parser.parse_args()

    shards = sorted(Path(args.data_dir).glob("train-*.tar"))
    if not shards:
        raise SystemExit(f"No shards found in {args.data_dir}")

    rng = random.Random(args.seed)
    shard = Path(args.data_dir) / f"{args.shard}.tar" if args.shard else rng.choice(shards)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"shard:   {shard.name} ({shard.stat().st_size / 1e6:.0f} MB)")
    print(f"samples: {save_dir}\n")

    with tarfile.open(shard) as tar:
        members = defaultdict(set)
        for m in tar.getmembers():
            p = Path(m.name)
            members[p.stem].add(p.suffix)

        incomplete = {k: v for k, v in members.items() if v != {".jpg", ".txt", ".json"}}
        keys = sorted(k for k, v in members.items() if v == {".jpg", ".txt", ".json"})
        print(f"{len(keys):,} complete samples, {len(incomplete)} incomplete")

        problems = defaultdict(list)
        boxes_total = boxes_zero = 0
        for key in keys:
            meta = json.loads(tar.extractfile(f"{key}.json").read())
            img = Image.open(io.BytesIO(tar.extractfile(f"{key}.jpg").read()))
            img.load()
            caption = tar.extractfile(f"{key}.txt").read().decode("utf-8")
            if caption != meta["caption"]:
                problems[".txt disagrees with meta caption"].append(key)
            boxes_total += len(meta["boxes_xywh"])
            boxes_zero += len(meta["boxes_xywh"]) == 0
            for p in check(key, img, meta):
                problems[p].append(key)

        print(f"{boxes_total / max(1, len(keys)):.2f} boxes/sample, "
              f"{boxes_zero} samples with no boxes")
        if problems:
            print("\nproblems:")
            for what, ks in sorted(problems.items(), key=lambda kv: -len(kv[1])):
                print(f"  {len(ks):>6}  {what}  (e.g. {ks[0]})")
        else:
            print("no structural problems found")

        print()
        for key in rng.sample(keys, min(args.num_samples, len(keys))):
            blob = tar.extractfile(f"{key}.jpg").read()
            meta = json.loads(tar.extractfile(f"{key}.json").read())
            img = Image.open(io.BytesIO(blob))
            img.load()
            (save_dir / f"{key}.jpg").write_bytes(blob)
            (save_dir / f"{key}.json").write_text(json.dumps(meta, indent=1) + "\n")
            draw_boxes(img, meta).save(save_dir / f"{key}_boxes.jpg", quality=95)

            print(f"key:      {key}")
            print(f"caption:  {meta['caption']}")
            print(f"image:    {img.size[0]}x{img.size[1]} from "
                  f"{meta['width']}x{meta['height']}, {len(blob) / 1024:.1f} KB")
            for (x, y, w, h), phrase in zip(meta["boxes_xywh"], meta["phrases"]):
                print(f"  box     [{x:.0f} {y:.0f} {w:.0f} {h:.0f}]  {phrase!r}")
            print()


if __name__ == "__main__":
    main()
