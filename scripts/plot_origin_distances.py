"""Embed grounded training samples and plot their distance from the origin.

One panel per model, one colour per supervision signal. The radius is what the
hyperbolic claim rests on -- general concepts should sit near the origin and
specific ones out towards the boundary -- so this draws the distribution each
model actually produces on the data it was trained on.

`--euclidean` reads the ablation twins instead, where the radius is the plain
norm and the boundary it would run out to does not exist. Each figure covers one
geometry: the panels stay in the same order so a pair can be read side by side.

Every model sees the same samples. Signals follow the model: `use_boxes=False`
trains on image and text alone, so plotting box radii for it would show a head
the loss never shaped.

The product-space models carry one radius per factor. They are combined here the
way the model's own metric combines them, so a PHyCLIP radius is on the same
footing as a MERU one.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gvlm import euclidean as E  # noqa: E402
from gvlm import lorentz as L  # noqa: E402
from gvlm.config import LazyConfig, instantiate  # noqa: E402
from gvlm.tokenizer import Tokenizer  # noqa: E402
from gvlm.utils.checkpointing import CheckpointManager  # noqa: E402
from gvlm.utils.directory import generate_output_dir_name  # noqa: E402

METHODS = ["meru", "hycoclip", "argent", "phyclip"]
DISPLAY = {
    "meru": "MERU",
    "hycoclip": "HyCoCLIP",
    "phyclip": "PHyCLIP",
    "argent": "ARGENT",
}

# Categorical slots 1, 2, 3, 7 of the house palette. Slot 4 collides with slot 2
# for full-colour readers (OKLab dE 13.7, below the floor of 15); violet clears
# every pair at 16.3 normal / 9.2 simulated.
SIGNALS = {
    "image": ("Image", "#2a78d6"),
    "text": ("Text", "#eb6834"),
    "box_image": ("Box image", "#1baf7a"),
    "box_text": ("Box text", "#4a3aa7"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


@dataclass(frozen=True)
class Relation:
    """A pair of signals with one pinned at `r1` and one swept over `r2`."""

    key: str
    parent: str
    child: str


# `oxy_angle(parent, child)` against the cone at `parent`, on the matched
# diagonal. Four terms, not one: the two cross relations take a box as the
# parent, and one of them never involves a text at all. There are no mirrors
# here -- the angle is read at the parent, so a flipped pair is a quantity no
# model computes.
ENTAILED = (
    Relation("text_image", "text", "image"),
    Relation("box", "box_text", "box_image"),
    Relation("cross_image", "box_image", "image"),
    Relation("cross_text", "box_text", "text"),
)

# `pairwise_dist` is only ever called cross-modally, and over the whole `B x B`
# grid, so the negatives are as much of the consumed region as the positives.
# It is symmetric, so which side is pinned is ours to choose and both appear.
CONTRASTED = (
    Relation("text_image", "text", "image"),
    Relation("image_text", "image", "text"),
    Relation("box", "box_text", "box_image"),
    Relation("box_flip", "box_image", "box_text"),
)


def signal_name(signal: str) -> str:
    return SIGNALS[signal][0].lower()


def supported(relations, signals) -> list[Relation]:
    return [r for r in relations if r.parent in signals and r.child in signals]


def load_model(method: str, arch: str, device, euclidean: bool = False):
    suffix = "_euclidean" if euclidean else ""
    config = Path("configs") / f"train_{method}{suffix}_vit_{arch}.py"
    cfg = LazyConfig.load(str(config))
    run = Path(generate_output_dir_name(cfg))
    checkpoint = run / "checkpoint_final.pth"
    if not checkpoint.is_file():
        return None, None, None
    model = instantiate(cfg.model).to(device).eval()
    CheckpointManager(model=model).load(str(checkpoint))
    # From the config, not the model: only PHyCLIP keeps use_boxes as an
    # attribute, so asking the model would silently drop every box signal.
    signals = ["image", "text"]
    if cfg.model.get("use_boxes", False):
        signals += ["box_image", "box_text"]
    return model, signals, run


def factors(model, feats: torch.Tensor) -> torch.Tensor:
    """`(B, D)` -> `(B, num_factors, factor_dim)`, one factor for the non-product models."""
    if hasattr(model, "num_subspaces"):
        return feats.view(-1, model.num_subspaces, model.subspace_dim)
    return feats.unsqueeze(1)


def combine(model, per_factor: torch.Tensor) -> torch.Tensor:
    """Collapse a per-factor reading over the last axis, as the product metric does."""
    if getattr(model, "product_metric", "l1") == "l1":
        return per_factor.mean(dim=-1)
    return per_factor.pow(2).mean(dim=-1).sqrt()


def origin_distance(model, feats: torch.Tensor) -> torch.Tensor:
    """Per-sample radius, mirroring the model's own `_origin_distances`."""
    subspaces = factors(model, feats)
    if hasattr(model, "curvs"):
        curvs = torch.stack([curv.detach().exp() for curv in model.curvs]).to(feats.device)
        per_factor = L.get_distance_from_origin(subspaces, curvs).squeeze(-1)
    elif hasattr(model, "curv"):
        curv = model.curv.detach().exp().to(feats.device)
        per_factor = L.get_distance_from_origin(subspaces, curv).squeeze(-1)
    else:
        per_factor = E.get_distance_from_origin(subspaces).squeeze(-1)
    return combine(model, per_factor)


def embed(model, signals, batches, device) -> dict[str, torch.Tensor]:
    tokenizer = Tokenizer()
    out = {s: [] for s in signals}
    with torch.inference_mode():
        for batch in batches:
            for signal in signals:
                if signal.endswith("image"):
                    feats = model.encode_image(batch[signal].to(device), project=True)
                else:
                    feats = model.encode_text(tokenizer(batch[signal]), project=True)
                out[signal].append(feats.float().cpu())
    return {s: torch.cat(v) for s, v in out.items()}


def collect(model, signals, batches, device) -> dict[str, np.ndarray]:
    return {
        s: origin_distance(model, f).numpy()
        for s, f in embed(model, signals, batches, device).items()
    }


def histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density and bin edges, binned per signal rather than per panel.

    Within a panel the spreads differ by up to 7x -- image radii are a spike
    next to box text -- so one shared bin width either smears the wide signal or
    collapses the narrow one to a single bar. Freedman-Diaconis on each signal's
    own spread resolves both; `density=True` keeps unit area under each, so the
    curves stay comparable despite the differing widths.
    """
    q25, q75 = np.percentile(values, [25, 75])
    width = 2 * (q75 - q25) / len(values) ** (1 / 3)
    span = values.max() - values.min()
    bins = 30 if width <= 0 else int(np.clip(round(span / width), 15, 120))
    return np.histogram(values, bins=bins, density=True)


def draw(results: dict[str, dict[str, np.ndarray]], out_path: Path):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Computer Modern, as on the distortion figure. cmr10 ships no
            # U+2212, so the hyphen-minus stands in or negative ticks box out.
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    )
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (method, dists) in zip(axes, results.items()):
        lo = min(d.min() for d in dists.values())
        hi = max(d.max() for d in dists.values())
        pad = 0.06 * (hi - lo or 1.0)

        for signal, (label, colour) in SIGNALS.items():
            if signal not in dists:
                continue
            density, edges = histogram(dists[signal])
            # Surface-coloured underlay: image and box image land within a few
            # thousandths of each other, and without it the two outlines merge.
            ax.stairs(density, edges, color="white", lw=2.6, zorder=3)
            ax.stairs(density, edges, color=colour, lw=1.0, label=label, zorder=4)
            ax.stairs(density, edges, color=colour, alpha=0.16, fill=True, lw=0, zorder=2)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(bottom=0)

        ax.set_title(DISPLAY.get(method, method), color=INK, pad=4)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=2)

    axes[0].set_ylabel("density")
    fig.supxlabel("distance from origin", fontsize=11, color=INK)
    handles = [
        plt.Line2D([], [], color=c, lw=1.6, label=lab) for lab, c in SIGNALS.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=len(SIGNALS),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        columnspacing=2.0,
    )
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--arch", default="s", choices=["s", "b"])
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--euclidean", action="store_true", help="Read the Euclidean ablation twins."
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Typeset with LaTeX in Computer Modern. Needs a LaTeX toolchain.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.usetex:
        plt.rcParams.update({"text.usetex": True})

    device = torch.device(args.device)

    # One fixed set of samples, held in memory, so every model is read on the
    # same images and captions rather than on its own draw from the shuffle.
    cfg = LazyConfig.load("configs/cc3m_dataset.py")
    cfg.dataset.infinite_stream = False
    cfg.dataset.seed = 0
    loader = torch.utils.data.DataLoader(
        instantiate(cfg.dataset), batch_size=args.batch_size, num_workers=4
    )
    batches, taken = [], 0
    for batch in loader:
        batches.append(batch)
        taken += batch["image"].shape[0]
        if taken >= args.num_samples:
            break
    print(f"{taken} samples over {len(batches)} batches")

    results = {}
    for method in args.methods:
        model, signals, run = load_model(method, args.arch, device, args.euclidean)
        if model is None:
            print(f"  skip {method}: no checkpoint_final.pth")
            continue
        results[method] = collect(model, signals, batches, device)
        summary = "  ".join(
            f"{s}={results[method][s].mean():.2f}" for s in signals
        )
        print(f"  {method:<10} {summary}")
        del model
        torch.cuda.empty_cache()

    if not results:
        raise SystemExit("no checkpoints found; nothing to plot")

    stem = "origin_distances_euclidean" if args.euclidean else "origin_distances"
    out = Path(args.out or f"outputs/{stem}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(results, out)

    # The table view the figure's colour encoding falls back to.
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "signal", "n", "mean", "std", "p10", "median", "p90"])
        for method, dists in results.items():
            for signal, d in dists.items():
                writer.writerow(
                    [method, signal, len(d)]
                    + [f"{v:.4f}" for v in (d.mean(), d.std(), *np.percentile(d, [10, 50, 90]))]
                )
    print(f"\n{out}  (+ .png, .csv)")


if __name__ == "__main__":
    main()
