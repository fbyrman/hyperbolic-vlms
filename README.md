# How Hyperbolic Are Hyperbolic Vision-Language Models?

Code for the paper *How Hyperbolic Are Hyperbolic Vision-Language Models?*,
accepted to the proceedings track of NeurReps 2026, the NeurIPS workshop on
Symmetry and Geometry in Neural Representations.

To help with understanding this work, we built an
[interactive tutorial](https://fbyrman.github.io/explainers/hyperbolic-vlms.html)
that focuses on the high-level intuition, with interactive plots to guide you
through it. All model weights are released on the Hugging Face Hub at
[`freek23/gvlm-checkpoints`](https://huggingface.co/freek23/gvlm-checkpoints),
one `<method>[_euclidean]_vit_{s,b}/model.pth` per training config.

## Layout

```
gvlm/                            model code: geometry, encoders, losses, evaluators
configs/                         train_<method>[_euclidean]_vit_{s,b}.py + eval_*.py
scripts/train.py                 training entry point
scripts/evaluate.py              evaluation entry point
scripts/plot_origin_distances.py radial distribution of each signal, per model
scripts/plot_distortion_ratios.py  distance, exterior angle and aperture distortion
scripts/extract_gligen_cc3m.py   GLIGEN TSVs -> webdataset tars + per-shard sidecars
scripts/build_manifest.py        sidecars -> manifest.json, checks consistency
scripts/sanity_check_cc3m.py     structural and visual check of one shard
```

`train_<method>_vit_b.py` holds the full config and the `_vit_s` variant
overrides only `model.visual.arch`. The dataset is defined once in
`configs/cc3m_dataset.py` and imported by all of them.

## Environment

```bash
uv venv ~/venvs/gvlm --python 3.10
uv pip install --python ~/venvs/gvlm/bin/python -r requirements-frozen.txt \
  --find-links https://download.pytorch.org/whl/cu124/torch \
  --find-links https://download.pytorch.org/whl/cu124/torchvision
uv pip install --python ~/venvs/gvlm/bin/python -e . --no-deps
```

`requirements-frozen.txt` is the exact environment the reported runs used, CUDA
12.4 wheels included. `pyproject.toml` holds the looser ranges if you need to
resolve against a different CUDA.

Every command below runs from the repository root.

## Data

Training needs CC3M images paired with GLIGEN's box annotations. Samples without
boxes are dropped at load time, so the grounded subset is the whole training set:
3,016,640 of CC3M's 3,318,333 rows.

```bash
python scripts/extract_gligen_cc3m.py --out-dir datasets/cc3m_grounded
python scripts/build_manifest.py --data-dir datasets/cc3m_grounded --verify
python scripts/sanity_check_cc3m.py --shard train-0000
```

Source is [`gligen/cc3m_tsv`](https://huggingface.co/datasets/gligen/cc3m_tsv),
pinned at revision `3f8d464e87669c5350787bdbc547f93dbb62357a` (483 TSV shards,
~518 GB). Those TSVs carry base64 JPEGs alongside the annotations, so no separate
image download is needed and none of the samples are lost to dead links; shards
are streamed over HTTP and never staged on disk. The extractor is idempotent,
counting a shard as done only once its `.stats.json` sidecar exists, so an
interrupted run is resumed by repeating the command, and `--shard-start` with
`--num-shards` splits the 483 shards over as many concurrent processes as you
care to give it. Tar members are written with tarfile's zeroed default metadata,
so a re-run at the same settings reproduces each shard byte for byte and the
recorded sha256 stays valid.

Result: 483 shards, 3,016,640 samples, 125 GB. Each sample is `.jpg` (RGB,
shortest side 256, quality 95), `.txt` (caption) and `.json` (caption, original
and resized dims, boxes, phrases, token spans), keyed by zero-padded `data_id`.

## Training

```bash
export CC3M_GROUNDED_DIR=$PWD/datasets/cc3m_grounded

python scripts/train.py --config configs/train_phyclip_vit_b.py --num-gpus 8
python scripts/train.py --config configs/train_phyclip_euclidean_vit_b.py --num-gpus 8
```

The matrix is `configs/train_<method>[_euclidean]_vit_{s,b}.py` over `meru`,
`hycoclip`, `phyclip` and `argent`. Any field can be overridden on the command
line, e.g. `train.total_batch_size=768`.

All runs are 100,000 iterations at `total_batch_size=512`. Batch size is global,
so `--num-gpus` changes throughput and not the optimization.

`--resume` is a no-op on a fresh run, so an interrupted run is resumed by
repeating the command unchanged. Each run writes to `train_results/<parameters>/`,
named after the config's own parameters rather than its filename.

## Evaluation

Zero-shot classification auto-downloads and caches most of its datasets in
`./datasets/eval`. ImageNet needs to be symlinked at `./datasets/eval/imagenet`
in [Torchvision ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
style, and Stanford Cars set up manually at `./datasets/eval/cars/stanford_cars`
following [pytorch/vision#7545](https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616).
Retrieval reads COCO captions and Flickr30k; see `gvlm/data/evaluation.py` for
how to arrange them.

```bash
python scripts/evaluate.py --config configs/eval_zero_shot_classification.py \
    --checkpoint-path train_results/<run>/checkpoint_final.pth \
    --train-config configs/train_phyclip_vit_b.py
```

To evaluate a released model instead, download it and pass it the same way:

```bash
hf download freek23/gvlm-checkpoints phyclip_vit_b/model.pth --local-dir checkpoints
python scripts/evaluate.py --config configs/eval_zero_shot_classification.py \
    --checkpoint-path checkpoints/phyclip_vit_b/model.pth \
    --train-config configs/train_phyclip_vit_b.py
```

Omitting `--config` runs classification, retrieval and hierarchical metrics in
turn. Results are appended to `evaluate.json` beside the checkpoint.

### Without the exponential map

The same weights scored in the tangent space instead of on the manifold: encode
with `project=False`, scale by the learned alpha, skip `exp_map0`, and score by
negative Euclidean distance. It asks how much of a hyperbolic model's accuracy
survives reading it flat.

```bash
python scripts/evaluate.py --config configs/eval_zero_shot_classification_ambient.py \
    --checkpoint-path train_results/<run>/checkpoint_final.pth \
    --train-config configs/train_phyclip_vit_b.py \
    --results-file evaluate_ambient.json
```

There is an `_ambient` config for each of the three tasks, each inheriting its
default counterpart's suite and prompts so the two runs stay comparable dataset
for dataset. The ablation is meaningless for the `Euclidean*` controls, which
have no exponential map to remove, and they fall back to their default scoring.

## Figures

### Radial distributions

Where each supervision signal sits relative to the origin, on the data the model
was trained on. The radius is what the hyperbolic claim rests on: general
concepts near the origin, specific ones out towards the boundary.

```bash
python scripts/plot_origin_distances.py --arch s
python scripts/plot_origin_distances.py --arch s --euclidean
```

One panel per model, one colour per signal, and a `.csv` beside the figure with
the per-signal quantiles. `--euclidean` reads the Euclidean controls, where the
radius is the plain norm; the panels keep their order so a pair reads side by
side. Product-space radii are combined the way the model's own metric combines
them, so a PHyCLIP radius is on the same footing as a MERU one.

### Distortion ratios

Distance, exterior angle and aperture, each as hyperbolic over Euclidean,
evaluated by the same functions the two arms of the ablation call. All three are
1 at the origin, so a panel says where in the ball the geometry starts doing work
flat space cannot.

```bash
python scripts/plot_distortion_ratios.py \
    --rows distance:text_image angle:text_image \
    --out outputs/distortion_main.pdf
```

A row is one quantity read on one relation -- a pair of signals with one pinned
at `r_1` and the other swept over `r_2`. `r_1`, the curvature and the marked
rectangle all come from the checkpoint, so the panel is the model's own reading
of the region its inputs actually occupy.

`--curv` plots stated geometries instead, one panel per curvature, with no
checkpoint loaded and no consumed region marked:

```bash
python scripts/plot_distortion_ratios.py --curv 0.1 0.5 1.0 2.0 \
    --r1 0.5 --r2-range 0 2 --rows distance angle aperture
```

This needs neither the data nor a GPU, and `--rows` may name a bare quantity,
there being no model to supply the relation. Use it for what the ratios do in
general; use the default for what a trained model does with them.

## Credit

The model code is built on the public code of three papers, and keeps their
structure (lazy configs, the webdataset training loop, the evaluation catalog):

- [MERU](https://github.com/facebookresearch/meru) -- Desai et al., *Hyperbolic Image-Text Representations*, ICML 2023
- [HyCoCLIP](https://github.com/PalAvik/hycoclip) -- Pal et al., *Compositional Entailment Learning for Hyperbolic Vision-Language Models*, ICLR 2025
- [PHyCLIP](https://github.com/tksmatsubara/PHyCLIP) -- Yoshikawa and Matsubara, *PHyCLIP: $\ell_1$-Product of Hyperbolic Factors Unifies Hierarchy and Compositionality in Vision-Language Representation Learning*, ICLR 2026

It is vendored from PHyCLIP at commit `0bf13d6` rather than submoduled, so the
repository clones as one piece. It differs from upstream in that the package was
renamed `phyclip` -> `gvlm`, training moved from GRIT to CC3M with GLIGEN
grounding, ViT-L configs were dropped, the VL-Checklist and SugarCrepe
evaluations were dropped, and ARGENT and the Euclidean controls were added.
