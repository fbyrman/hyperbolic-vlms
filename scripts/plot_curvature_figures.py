"""Plot the three curvature ratios over the region each model actually consumes.

Every ratio is hyperbolic over Euclidean, evaluated by the same functions the two
arms of the ablation call, so a panel is the pair's own reading of one
configuration rather than a restatement of either arm:

    distortion  geodesic distance over the tangent-space chord
    angle       `L.oxy_angle` over `E.oxy_angle` at the same apex
    aperture    `L.half_aperture` over `E.half_aperture` at the twin's 2K/sqrt(c)

All three are 1 at the origin, so a panel says where in the ball the geometry
starts doing work flat space cannot.

A row is one quantity read on one relation -- a pair of signals with one pinned
at `r_1` and the other swept over `r_2`. The relation is the unit the losses
consume, and there are four for the grounded models, not one, so `r_1` is never
pooled across relations. Panels label the axes `r_1` and `r_2` only; which
signals those are is fixed by the row and belongs in the caption. The output
filename names them.

The angle is read at the apex, so its relations are directed and have no
mirrors. The distortion ratio is symmetric, so the pin is a display choice and
both pins are offered. The aperture depends on the apex radius alone, so it
takes apexes rather than pairs and is a curve rather than a surface.

The marked rectangle is where the model's own inputs land: the whole `B x B`
grid for the distortion ratio, whose contrastive term reads every negative, and
the matched diagonal for the angle ratio, whose entailment term reads only
positives. It is the 1st-to-99th percentile of `r_2` crossed with the same range
of `theta`, so it bounds the pairs rather than holding a stated fraction of them.

    python scripts/plot_curvature_figures.py \\
        --rows distortion:text_image angle:text_image \\
        --out outputs/curvature_main.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gvlm import euclidean as E  # noqa: E402
from gvlm import lorentz as L  # noqa: E402
from gvlm.config import LazyConfig, instantiate  # noqa: E402
from gvlm.models import ARGENT  # noqa: E402
from plot_origin_distances import CONTRASTED, DISPLAY, ENTAILED, GRID, INK  # noqa: E402
from plot_origin_distances import METHODS, MUTED, SIGNALS  # noqa: E402
from plot_origin_distances import combine, embed, factors, load_model  # noqa: E402
from plot_origin_distances import origin_distance, supported  # noqa: E402

# The blue and orange of the origin-distance figure as diverging poles, over the
# neutral the house pair uses. Cool to warm through a grey that reads as
# "nothing", so the midpoint sits where curvature stops mattering.
CMAP = LinearSegmentedColormap.from_list(
    "gvlm_div_blue_orange",
    ["#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f5b483", "#eb6834", "#c2461a"],
)

# The aperture ratio cannot exceed 1, so it only ever occupies the cool half.
# Handing it the full diverging map would put its no-effect plateau at the warm
# pole and read as the opposite of what it is: this is that half, alone. A ratio
# of 1 starts at the house blue rather than at a near-white, because almost every
# apex sits at exactly 1 and a panel ramping to the neutral reads as blank paper
# rather than as a value.
CMAP_COOL = LinearSegmentedColormap.from_list(
    "gvlm_seq_blue", ["#0a2f5c", "#12508f", "#2a78d6", "#5b9be8", "#86b6ef"]
)

# ARGENT reads the angle but never forms a cone.
CONELESS = (ARGENT,)

def get_distortion_ratio(r1, r2, c, theta):
    sqrt_c = np.sqrt(c)
    xi1, xi2 = sqrt_c * r1, sqrt_c * r2
    arg = np.cosh(xi1) * np.cosh(xi2) - np.sinh(xi1) * np.sinh(xi2) * np.cos(theta)
    num = np.arccosh(np.maximum(arg, 1.0 + 1e-15))
    dist_sq = xi1**2 + xi2**2 - 2 * xi1 * xi2 * np.cos(theta)
    ratio = num / np.sqrt(np.maximum(dist_sq, 1e-15))
    mask = dist_sq < 1e-10  # theta->0 / coincident-point limit
    if np.any(mask):
        xi = (xi1 + xi2) / 2.0
        limit = np.where(xi < 1e-5, 1.0 + xi**2 / 6.0, np.sinh(xi) / xi)
        ratio = np.where(mask, limit, ratio)
    return ratio


def space_norm(r, c):
    """Geodesic radius to space-component norm; `lorentz` takes the latter."""
    sqrt_c = np.sqrt(c)
    return np.sinh(sqrt_c * r) / sqrt_c


def _pair(n1, n2, theta):
    x = np.stack([np.broadcast_to(n1, theta.shape), np.zeros_like(theta)], -1)
    y = np.stack([n2 * np.cos(theta), n2 * np.sin(theta)], -1)
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def get_exterior_angle(r1, r2, c, theta):
    x, y = _pair(space_norm(r1, c), space_norm(r2, c), theta)
    return L.oxy_angle(x, y, curv=c).numpy()


def get_flat_exterior_angle(r1, r2, theta):
    """The same pair before `exp_map0`, where the radii are the tangent norms."""
    x, y = _pair(r1, np.broadcast_to(r2, theta.shape), theta)
    return E.oxy_angle(x, y).numpy()


def get_angle_ratio(r1, r2, c, theta, tol: float = 1e-6):
    ratio = get_exterior_angle(r1, r2, c, theta) / np.maximum(
        get_flat_exterior_angle(r1, r2, theta), 1e-15
    )
    # Both angles vanish as theta -> 0 with the far point outside, and both run
    # to pi with it inside, so the quotient is 0/0 on one half of that edge and
    # 1 on the other. The law of sines gives the first limit outright.
    xi1, xi2 = np.sqrt(c) * r1, np.sqrt(c) * np.broadcast_to(r2, theta.shape)
    gap = xi2 - xi1
    outward = gap > tol
    # Both factors of the denominator are guarded, not just `gap`: `np.where`
    # evaluates the branch it discards, and an analytic window may start at
    # `r2 = 0`, where `xi2` vanishes and the quotient warns before being thrown
    # away.
    denom = np.where(outward, xi2, 1.0) * np.sinh(np.where(outward, gap, 1.0))
    limit = np.where(outward, np.sinh(xi2) * gap / denom, 1.0)
    return np.where(theta < tol, limit, ratio)


def clamp_radius(c: float, min_radius: float) -> float:
    """Below this the asin argument exceeds 1 and the aperture is left undefined."""
    return float(np.arcsinh(2 * min_radius) / np.sqrt(c))


# `lorentz` bounds its asin at 1 - 1e-8 and `euclidean` at 1 - 1e-6, and inside
# the clamped neighbourhood that gap is the entire quotient: it puts the plateau
# at 1 + 8e-4 and reads as curvature widening a cone it has not touched. Both
# arms are held to the same bound here so the plateau is exactly 1. This is the
# one place the figure departs from the arms' own code, and it only moves the
# epsilon -- the un-clamping mismatch below is unaffected, since `2K/sinh(xi)`
# reaches 1 at `xi = asinh(2K)` and `2K/xi` at `xi = 2K` whatever the bound is.
_ASIN_EPS = 1e-8


def get_half_aperture(r, c, min_radius, eps: float = _ASIN_EPS):
    n = np.atleast_1d(space_norm(np.asarray(r, dtype=np.float64), c))
    x = torch.from_numpy(np.stack([n, np.zeros_like(n)], -1))
    return (
        L.half_aperture(x, curv=c, min_radius=min_radius, eps=eps)
        .numpy()
        .reshape(np.shape(r))
    )


def get_flat_half_aperture(r, c, min_radius, eps: float = _ASIN_EPS):
    """The same apex before `exp_map0`, at the twin's `aperture_k = 2K/sqrt(c)`.

    Written out rather than calling `E.half_aperture`, which is fixed to the
    coarser bound.
    """
    n = np.asarray(r, dtype=np.float64)
    k = 2 * min_radius / np.sqrt(c)
    return np.arcsin(np.clip(k / (n + eps), -1.0 + eps, 1.0 - eps))


def get_aperture_ratio(r, c, min_radius):
    """The 0.7 and 1.2 thresholds cancel, being applied outside both clamps."""
    return get_half_aperture(r, c, min_radius) / get_flat_half_aperture(r, c, min_radius)


def get_aperture_surface(r1, r2, c, theta, min_radius=0.1):
    """The aperture over the same plane as its two siblings.

    It is a function of the apex radius alone, so `r2` here carries the apex and
    the field is flat in `theta` -- which is the reading, not a defect: the cone
    the loss compares an angle against does not know the angle.
    """
    return np.broadcast_to(get_aperture_ratio(r2, c, min_radius), np.shape(theta)).copy()


# Surface, colourbar label, the relations it is defined on, how the loss pairs
# them up, which end of the relation is swept up the y axis, and that axis' name.
SURFACES = {
    "distortion": (
        get_distortion_ratio, r"$r_d$", CONTRASTED, "all_pairs", "child", r"$r_2$", CMAP,
    ),
    "angle": (
        get_angle_ratio, r"$r_{\mathrm{ext}}$", ENTAILED, "diagonal", "child", r"$r_2$",
        CMAP,
    ),
    "aperture": (
        get_aperture_surface, r"$r_{\mathrm{aper}}$", ENTAILED, "diagonal", "parent",
        r"$r_1$", CMAP_COOL,
    ),
}


def model_curvature(model) -> float:
    """The trained curvature, averaged over factors for the product models.

    Matching how their radii and angles are combined, so `r1`, `r2`, `theta` and
    `c` all describe the same collapsed space rather than four readings of it.
    """
    if hasattr(model, "curvs"):
        return float(torch.stack([curv.exp() for curv in model.curvs]).mean())
    return float(model.curv.exp())


def origin_angles(model, parent, child, pairing, budget: int = 1 << 22) -> np.ndarray:
    """The angle at the origin subtended by each consumed pair.

    This is the `theta` the surfaces sweep: the geodesic from the origin leaves
    along the space component, so the angle between two of them is the angle at
    `O` in the triangle `Oxy`.
    """
    p, c = factors(model, parent), factors(model, child)
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if pairing == "diagonal":
        return combine(model, (p * c).sum(-1).clamp(-1, 1).arccos()).numpy()
    step = max(1, budget // (c.shape[0] * c.shape[1]))
    out = [
        combine(model, torch.einsum("pnd,cnd->pcn", p[i : i + step], c).clamp(-1, 1).arccos())
        for i in range(0, p.shape[0], step)
    ]
    return torch.cat([o.flatten() for o in out]).numpy()


def mode(x: np.ndarray, resolution: int = 512) -> float:
    """The peak of the marginal, not its mean: pooling or averaging a skewed
    radius distribution puts `r_1` where the model has no mass."""
    grid = np.linspace(x.min(), x.max(), resolution)
    return float(grid[np.argmax(gaussian_kde(x)(grid))])


# One format for every ratio and one for every radius, so no two panels invite
# a comparison at different precision.
RATIO_FMT, RADIUS_FMT = "%.3f", "%.2f"


def occupancy(model, feats, relation, pairing, sweep, spread, min_radius) -> dict:
    swept = getattr(relation, sweep)
    y = origin_distance(model, feats[swept]).numpy()
    theta = origin_angles(model, feats[relation.parent], feats[relation.child], pairing)
    y_lo, y_hi = np.percentile(y, spread)
    th_lo, th_hi = np.percentile(theta, spread)
    curv = model_curvature(model)
    return {
        "r1": mode(origin_distance(model, feats[relation.parent]).numpy()),
        "y_lo": float(y_lo),
        "y_hi": float(y_hi),
        "th_lo": float(th_lo),
        "th_hi": float(th_hi),
        "c": curv,
        "r_min": clamp_radius(curv, min_radius),
    }


def analytic_panel(curv: float, r1: float, min_radius: float) -> dict:
    """A panel from a stated geometry rather than from a checkpoint's embeddings.

    Same surfaces, no occupancy: `r1` and `c` are given rather than read off a
    model, so nothing marks a consumed region and the panel is the ratio over
    the whole window.
    """
    return {"r1": r1, "c": curv, "r_min": clamp_radius(curv, min_radius)}


def row_window(panels: dict, quantity: str, override=None) -> tuple[float, float]:
    if override:
        return tuple(override)
    if any("y_lo" not in p for p in panels.values() if p is not None):
        raise SystemExit("--r2-range is required without a model to read it from")
    drawn = [p for p in panels.values() if p is not None]
    lo = min(p["y_lo"] for p in drawn)
    hi = max(p["y_hi"] for p in drawn)
    pad = 0.05 * (hi - lo)
    return float(np.floor((lo - pad) * 20) / 20), float(np.ceil((hi + pad) * 20) / 20)


def row_surfaces(quantity, panels, window, resolution):
    surface = SURFACES[quantity][0]
    Theta, R2 = np.meshgrid(
        np.linspace(0.0, np.pi, resolution), np.linspace(*window, resolution)
    )
    drawn = {m: p for m, p in panels.items() if p is not None}
    return Theta, R2, {m: surface(p["r1"], R2, p["c"], Theta) for m, p in drawn.items()}


def draw_surface_row(axes, cax, methods, quantity, panels, window, grids, limits):
    Theta, R2, surfaces = grids
    (vmin, vmax), flat = limits

    mesh = None
    for ax, method in zip(axes, methods):
        p = panels.get(method)
        if p is None:
            missing = "no entailment cone" if quantity == "aperture" else "relation not trained"
            ax.text(
                0.5, 0.5, missing, transform=ax.transAxes, ha="center",
                va="center", color=MUTED, fontsize=8,
            )
            ax.set_axis_off()
            continue
        mesh = ax.pcolormesh(
            Theta, R2, surfaces[method], cmap=SURFACES[quantity][6], shading="auto",
            vmin=vmin, vmax=vmax, rasterized=True,
        )
        # Where this model's own inputs to this quantity land. Marking only the
        # radii would imply every angle is consumed equally, and the pairs
        # concentrate far too tightly for that to be honest.
        if "y_lo" in p:
            ax.add_patch(
                Rectangle(
                    (p["th_lo"], p["y_lo"]),
                    p["th_hi"] - p["th_lo"],
                    p["y_hi"] - p["y_lo"],
                    fill=False, edgecolor=INK, lw=0.7, ls=(0, (3, 2)), alpha=0.65, zorder=3,
                )
            )
        if SURFACES[quantity][4] == "child":
            # Bottom right: the consumed box sits at low theta in every model,
            # so this is the one corner it cannot collide with.
            ax.annotate(
                f"$r_1 = {p['r1']:.2f}$",
                xy=(0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
                zorder=5,
            )
        ax.set_xlim(0.0, np.pi)
        ax.set_ylim(*window)
        ax.set_xticks([0, np.pi / 2, np.pi])
        ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$"])
        if method == methods[0]:
            ax.yaxis.set_major_formatter(FormatStrFormatter(RADIUS_FMT))

    bar = cax.figure.colorbar(mesh, cax=cax)
    bar.set_label(SURFACES[quantity][1], fontsize=13, color=INK)
    # A flat quantity has no range to tick; one label says what it is worth.
    bar.set_ticks([vmax] if flat else np.linspace(vmin, vmax, 5))
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter(RATIO_FMT))
    bar.ax.tick_params(labelsize=9, colors=MUTED, length=2)
    bar.outline.set_visible(False)


def draw(rows, methods, out_path: Path, resolution: int, r2_range, min_radius):
    nrow, ncol = len(rows), len(methods)
    fig = plt.figure(figsize=(2.15 * ncol + 0.75, 2.6 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1.0] * ncol + [0.055])

    windows = [row_window(panels, q, r2_range) for q, _, panels in rows]
    grids = [
        row_surfaces(q, panels, w, resolution)
        for (q, _, panels), w in zip(rows, windows)
    ]
    # One colour scale per quantity across the whole figure: the same ratio must
    # not be pale in one row and saturated in the next.
    limits = {}
    for (quantity, _, _), (_, _, surfaces) in zip(rows, grids):
        lo = min([limits.get(quantity, (np.inf, -np.inf))[0]]
                 + [float(Z.min()) for Z in surfaces.values()])
        hi = max([limits.get(quantity, (np.inf, -np.inf))[1]]
                 + [float(Z.max()) for Z in surfaces.values()])
        limits[quantity] = (lo, hi)
    for quantity, (lo, hi) in limits.items():
        flat = hi - lo < 1e-9  # a quantity constant everywhere it is read
        limits[quantity] = ((lo - 1e-3, hi) if flat else (lo, hi), flat)

    for i, (quantity, spec, panels) in enumerate(rows):
        axes = [fig.add_subplot(gs[i, j]) for j in range(ncol)]
        window = windows[i]
        cax = fig.add_subplot(gs[i, ncol])
        draw_surface_row(
            axes, cax, methods, quantity, panels, window, grids[i], limits[quantity]
        )
        ylabel = SURFACES[quantity][5]

        for ax, method in zip(axes, methods):
            if not ax.axison:
                continue
            if i == 0:
                p = panels.get(method)
                title = DISPLAY.get(method, method)
                if p is not None and "y_lo" in p:
                    title += f"\n$c = {p['c']:.2f}$"
                ax.set_title(title, color=INK, pad=4, fontsize=9)
            if method == methods[0]:
                ax.set_ylabel(ylabel, fontsize=12, color=INK)
            else:
                ax.set_yticklabels([])
            if i == nrow - 1:
                ax.set_xlabel(r"$\theta$", fontsize=11, color=INK)
            else:
                ax.set_xticklabels([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, length=2)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def parse_row(spec: str, needs_relation: bool = True):
    """`quantity:relation`, or a bare quantity where no model supplies the pair."""
    quantity, _, key = spec.partition(":")
    if quantity not in SURFACES:
        raise SystemExit(f"unknown quantity {quantity!r}; pick from {list(SURFACES)}")
    if not key and not needs_relation:
        return quantity, None
    catalog = {r.key: r for r in SURFACES[quantity][2]}
    if key not in catalog:
        raise SystemExit(f"{quantity} is not defined on {key!r}; pick from {list(catalog)}")
    return quantity, catalog[key]


def row_key(quantity, relation) -> str:
    return quantity if relation is None else f"{quantity}:{relation.key}"


def cached_embed(method, arch, model, signals, batches, device, cache_dir, run):
    """Embeddings are the whole cost of these figures and never change for a
    finished checkpoint, so a run that only reshapes a panel pays nothing."""
    if cache_dir is None:
        return embed(model, signals, batches, device)
    taken = sum(b["image"].shape[0] for b in batches)
    stamp = int((Path(run) / "checkpoint_final.pth").stat().st_mtime)
    path = Path(cache_dir) / f"{method}_{arch}_{taken}_{stamp}.pt"
    if path.is_file():
        blob = torch.load(path)
        if all(s in blob for s in signals):
            print(f"    cache hit {path.name}")
            return blob
    feats = embed(model, signals, batches, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats


def finish(cells, wanted, methods, args):
    rows = []
    for quantity, spec in wanted:
        panels = cells[row_key(quantity, spec)]
        if all(p is None for p in panels.values()):
            print(f"  drop row {row_key(quantity, spec)}: no model has it")
            continue
        rows.append((quantity, spec, panels))
    if not rows:
        raise SystemExit("no row survives; nothing to plot")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(rows, methods, out, args.resolution, args.r2_range, args.min_radius)
    print(f"\n{out}  (+ .png)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument(
        "--rows",
        nargs="+",
        default=["distortion:text_image", "angle:text_image"],
        metavar="QUANTITY:RELATION",
        help="One row per entry. distortion: text_image, image_text, box, "
             "box_flip. angle and aperture: text_image, box, cross_image, "
             "cross_text.",
    )
    parser.add_argument(
        "--curv",
        type=float,
        nargs="+",
        metavar="C",
        help="Plot stated geometries instead of trained ones: one panel per "
             "curvature, read at --r1 over --r2-range, with no checkpoint "
             "loaded and no consumed region marked. --rows may then name a "
             "bare quantity, there being no model to supply the relation.",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.5,
        help="The pinned radius, with --curv. Read off the model otherwise.",
    )
    parser.add_argument("--arch", default="s", choices=["s", "b"])
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--spread",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LO", "HI"),
        help="Percentiles of the consumed radii and angles that set the marked box.",
    )
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument(
        "--min-radius", type=float, default=0.1, help="The half-aperture clamp radius."
    )
    parser.add_argument(
        "--r2-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="Shared r2 window for every surface row. Default is per row.",
    )
    parser.add_argument("--cache-dir", default="outputs/.embed_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="outputs/curvature_figure.pdf")
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Typeset with LaTeX in Computer Modern. Needs a LaTeX toolchain.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Computer Modern without a LaTeX toolchain: mathtext draws the same
            # glyphs LaTeX would for the symbols here. cmr10 ships no U+2212, so
            # the hyphen-minus has to stand in or every negative tick is a box.
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    )
    if args.usetex:
        plt.rcParams.update({"text.usetex": True})

    wanted = [parse_row(spec, needs_relation=not args.curv) for spec in args.rows]
    cells = {row_key(q, s): {} for q, s in wanted}

    if args.curv:
        methods = [f"$c = {c:g}$" for c in args.curv]
        if len(set(methods)) != len(methods):
            raise SystemExit("--curv values must stay distinct once formatted")
        for label, curv in zip(methods, args.curv):
            panel = analytic_panel(curv, args.r1, args.min_radius)
            for quantity, relation in wanted:
                cells[row_key(quantity, relation)][label] = panel
            print(f"  c={curv:<8g} r1={args.r1:.3f}  r_min={panel['r_min']:.3f}")
        return finish(cells, wanted, methods, args)

    cache_dir = None if args.no_cache else args.cache_dir
    device = torch.device(args.device)

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

    methods = []
    for method in args.methods:
        model, signals, run = load_model(method, args.arch, device)
        if model is None:
            print(f"  skip {method}: no checkpoint_final.pth")
            continue
        methods.append(method)
        print(f"  {method:<10} c={model_curvature(model):.3f}")
        feats = cached_embed(
            method, args.arch, model, signals, batches, device, cache_dir, run
        )
        for quantity, relation in wanted:
            key = row_key(quantity, relation)
            coneless = quantity == "aperture" and isinstance(model, CONELESS)
            if coneless or not supported([relation], signals):
                cells[key][method] = None
                print(f"    {key:<26} {'no cone' if coneless else 'not trained'}")
                continue
            _, _, _, pairing, sweep, _, _ = SURFACES[quantity]
            p = occupancy(
                model, feats, relation, pairing, sweep, args.spread, args.min_radius
            )
            cells[key][method] = p
            print(
                f"    {key:<26} r1={p['r1']:.3f}  "
                f"y=[{p['y_lo']:.3f}, {p['y_hi']:.3f}]  "
                f"theta=[{p['th_lo']:.3f}, {p['th_hi']:.3f}]"
            )
        del model, feats
        torch.cuda.empty_cache()

    if not methods:
        raise SystemExit("no checkpoints found; nothing to plot")

    return finish(cells, wanted, methods, args)


if __name__ == "__main__":
    main()
