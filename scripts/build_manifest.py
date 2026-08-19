#!/usr/bin/env python
"""Aggregate per-shard sidecars into a dataset manifest.

Checks every shard is present and built with identical settings, then writes
manifest.json with per-shard checksums. --verify re-hashes the tars on disk,
which reads the whole dataset, so it is off by default.
"""
import argparse
import hashlib
import json
from pathlib import Path

NUM_SHARDS = 483


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    shards, missing, mismatched = [], [], []
    settings = None

    for i in range(NUM_SHARDS):
        name = f"train-{i:04d}"
        sidecar = data_dir / f"{name}.stats.json"
        tar = data_dir / f"{name}.tar"
        if not sidecar.exists() or not tar.exists():
            missing.append(i)
            continue

        stats = json.loads(sidecar.read_text())
        if settings is None:
            settings = stats["settings"]
        elif stats["settings"] != settings:
            mismatched.append((i, stats["settings"]))

        size = tar.stat().st_size
        if size != stats["bytes"]:
            mismatched.append((i, f"size {size} != recorded {stats['bytes']}"))
        if args.verify:
            digest = sha256_file(tar)
            if digest != stats["sha256"]:
                mismatched.append((i, f"sha256 {digest[:12]} != recorded {stats['sha256'][:12]}"))
        shards.append(stats)

    if missing:
        rng = ", ".join(str(i) for i in missing[:20]) + (" ..." if len(missing) > 20 else "")
        print(f"{len(missing)} shard(s) missing: {rng}")
    if mismatched:
        print(f"{len(mismatched)} shard(s) inconsistent:")
        for idx, why in mismatched[:20]:
            print(f"  {idx:04d}: {why}")
    if missing or mismatched:
        raise SystemExit(1)

    total_samples = sum(s["samples"] for s in shards)
    total_bytes = sum(s["bytes"] for s in shards)
    source = shards[0]["source"]

    manifest = {
        "source": {"repo_id": source["repo_id"], "revision": source["revision"]},
        "settings": settings,
        "num_shards": len(shards),
        "num_samples": total_samples,
        "num_unusable_rows": sum(s["unusable_rows"] for s in shards),
        "bytes": total_bytes,
        "shards": [
            {"shard": s["shard"], "samples": s["samples"], "bytes": s["bytes"],
             "sha256": s["sha256"]}
            for s in shards
        ],
    }

    out = data_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{len(shards)} shards, {total_samples:,} samples, {total_bytes / 1e9:.1f} GB")
    print(f"settings: {settings}   source: {source['repo_id']}@{source['revision'][:8]}")
    print(f"wrote {out}" + ("" if args.verify else "  (re-run with --verify to check sha256)"))


if __name__ == "__main__":
    main()
