#!/usr/bin/env python
"""Stream gligen/cc3m_tsv and write image+caption+grounding webdataset shards.

Each source TSV row is `<line_idx>\\t<json>`, where the JSON holds the base64
JPEG under "image" and per-box annotations under "annos". The published TSVs
also carry precomputed CLIP embeddings (~40% of each row) which are dropped.

Boxes are XYWH in original-image pixel coordinates. Both the original-pixel
boxes and a resolution-independent normalized copy are written, so downstream
code never has to reconstruct the resize factor.

Shards are streamed over HTTP and never staged on disk. Output shards are
written to a .tmp path and renamed on completion, so an interrupted run is
resumed by re-invoking with the same arguments. Each finished shard gets a
.stats.json sidecar with its sha256 and sample count.
"""
import argparse
import base64
import hashlib
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

import requests
from PIL import Image

REPO_ID = "gligen/cc3m_tsv"
REVISION = "3f8d464e87669c5350787bdbc547f93dbb62357a"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
NUM_SHARDS = 483


class HashingWriter:
    def __init__(self, fileobj):
        self._f = fileobj
        self._h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data):
        self._h.update(data)
        self.bytes_written += len(data)
        return self._f.write(data)

    def tell(self):
        return self._f.tell()

    def flush(self):
        return self._f.flush()

    @property
    def hexdigest(self):
        return self._h.hexdigest()


def phrases_for(caption, tokens_positive):
    return " ".join(caption[s:e] for s, e in tokens_positive)


def resize_shortest(img, target):
    w, h = img.size
    shortest = min(w, h)
    if shortest <= target:
        return img
    scale = target / shortest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def build_sample(rec, image_size, quality):
    img = Image.open(io.BytesIO(base64.b64decode(rec["image"])))
    img.load()
    ow, oh = img.size

    img = resize_shortest(img.convert("RGB"), image_size)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)

    caption = rec.get("caption", "")
    boxes, tokens, phrases = [], [], []
    for anno in rec.get("annos", []):
        bbox = anno.get("bbox")
        if bbox is None:
            continue
        tp = anno.get("tokens_positive", [])
        boxes.append(bbox)
        tokens.append(tp)
        phrases.append(phrases_for(caption, tp))

    meta = {
        "data_id": rec.get("data_id"),
        "file_name": rec.get("file_name"),
        "caption": caption,
        "width": ow,
        "height": oh,
        "resized_width": img.size[0],
        "resized_height": img.size[1],
        "boxes_xywh": boxes,
        "boxes_xywh_norm": [[x / ow, y / oh, w / ow, h / oh] for x, y, w, h in boxes],
        "tokens_positive": tokens,
        "phrases": phrases,
    }
    return f"{rec['data_id']:012d}", buf.getvalue(), caption, meta


def process_shard(task):
    shard_idx, out_dir, image_size, quality, retries = task
    name = f"train-{shard_idx:04d}"
    final = Path(out_dir) / f"{name}.tar"
    sidecar = Path(out_dir) / f"{name}.stats.json"

    # Sidecar, not tar, marks completion: a tar left by a killed worker is redone.
    if final.exists() and sidecar.exists():
        return shard_idx, "skipped", 0, 0

    tmp = Path(out_dir) / f"{name}.tar.tmp"
    url = f"{REPO_URL}/{name}.tsv"

    for attempt in range(retries + 1):
        written = unusable = 0
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as raw:
                    writer = HashingWriter(raw)
                    with tarfile.open(fileobj=writer, mode="w") as tar:
                        for line in resp.iter_lines(chunk_size=4 << 20):
                            if not line:
                                continue
                            try:
                                _, _, payload = line.partition(b"\t")
                                key, jpeg, caption, meta = build_sample(
                                    json.loads(payload), image_size, quality
                                )
                            except Exception:
                                unusable += 1
                                continue
                            for ext, blob in (
                                (".jpg", jpeg),
                                (".txt", caption.encode("utf-8")),
                                (".json", json.dumps(meta).encode("utf-8")),
                            ):
                                info = tarfile.TarInfo(key + ext)
                                info.size = len(blob)
                                tar.addfile(info, io.BytesIO(blob))
                            written += 1
                    digest, nbytes = writer.hexdigest, writer.bytes_written

            os.replace(tmp, final)
            sidecar.write_text(json.dumps({
                "shard": name,
                "samples": written,
                "unusable_rows": unusable,
                "bytes": nbytes,
                "sha256": digest,
                "source": {"repo_id": REPO_ID, "revision": REVISION, "file": f"{name}.tsv"},
                "settings": {"image_size": image_size, "quality": quality},
            }, indent=1) + "\n")
            return shard_idx, "ok", written, unusable
        except Exception as exc:
            Path(tmp).unlink(missing_ok=True)
            if attempt == retries:
                return shard_idx, f"FAILED: {type(exc).__name__}: {exc}", 0, 0
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = range(args.shard_start, min(args.shard_start + args.num_shards, NUM_SHARDS))
    if not shards:
        print(f"nothing to do: shard-start {args.shard_start} is past {NUM_SHARDS}", flush=True)
        return
    tasks = [(i, str(out_dir), args.image_size, args.quality, args.retries) for i in shards]

    print(f"{len(tasks)} shards [{shards[0]:04d}..{shards[-1]:04d}] -> {out_dir} "
          f"at {args.image_size}px q{args.quality}, {args.workers} workers, "
          f"source {REPO_ID}@{REVISION[:8]}", flush=True)

    start = time.time()
    done = total_written = total_unusable = 0
    failures = []
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        for idx, status, written, unusable in pool.imap_unordered(process_shard, tasks):
            done += 1
            total_written += written
            total_unusable += unusable
            if status.startswith("FAILED"):
                failures.append((idx, status))
            elapsed = time.time() - start
            rate = total_written / elapsed if elapsed else 0
            print(f"[{done}/{len(tasks)}] shard {idx:04d} {status} "
                  f"| {total_written:,} imgs | {rate:.0f} img/s | {elapsed/60:.1f} min",
                  flush=True)

    print(f"\ndone: {total_written:,} images written, {total_unusable:,} rows unusable, "
          f"in {(time.time()-start)/60:.1f} min", flush=True)
    if failures:
        print(f"{len(failures)} shard(s) failed:", flush=True)
        for idx, status in failures:
            print(f"  {idx:04d}: {status}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
