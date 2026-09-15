#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from _common import output_dir, provenance

MM = 1.0 / 25.4

METHOD_ORDER = ["GenPerturb", "rE2G_extended", "rE2G", "ABC", "TSS_distance"]
METHOD_LABELS = ["GenPerturb", "rE2G ext.", "rE2G", "ABC", "TSS distance"]
METHOD_COLORS = {"GenPerturb": "#4C92C3", "rE2G_extended": "#9B79C6", "rE2G": "#4CAF50",
                 "ABC": "#FF8C32", "TSS_distance": "#999999"}
STRATA = ["promoter_0_1kb", "proximal_1_10kb", "distal_10_100kb", "very_distal_100kb"]
STRATA_LABELS = ["0-1 kb (promoter)", "1-10 kb (proximal)",
                 "10-100 kb (distal)", ">=100 kb (very distal)"]
STRATA_BOUNDS = {"promoter_0_1kb": (0, 1000), "proximal_1_10kb": (1000, 10000),
                 "distal_10_100kb": (10000, 100000), "very_distal_100kb": (100000, np.inf)}
PERTURBATIONS = ["GFI1B", "MYB", "GATA1", "TAL1", "NFE2", "RUNX1", "LMO2"]

# Type sizes in points; these are the sizes that end up on the page.
FS_TICK = 5.5
FS_TITLE = 6.0
FS_LABEL = 6.0
FS_LEGEND = 5.5


def style() -> None:
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.size": FS_TICK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.6,
        "ytick.major.size": 1.6,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "legend.fontsize": FS_LEGEND,
        "lines.linewidth": 0.7,
    })


def add_axes_mm(fig, canvas_w, canvas_h, x, y, w, h):
    """Axes placed by millimetre box measured from the top-left of the canvas."""
    return fig.add_axes([x / canvas_w, (canvas_h - y - h) / canvas_h,
                         w / canvas_w, h / canvas_h])


def new_fig(canvas_w, canvas_h):
    return plt.figure(figsize=(canvas_w * MM, canvas_h * MM))


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Transparent backgrounds support placement in the composite figure.
    fig.savefig(out_dir / f"{name}.svg", transparent=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_dir / name}.svg")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_auprc(results_root: Path) -> pd.DataFrame:
    df = pd.read_csv(results_root / "output/auprc_per_perturbation.tsv", sep="\t")
    df = df[df.universe == "independent"].copy()
    df["perturbation"] = df.perturbation.str.removeprefix("MartinRufino.")
    return df


def eligible_table(df: pd.DataFrame) -> pd.DataFrame:
    """Perturbation x method AUPRC, keeping only perturbations scored by every method."""
    wide = df.pivot(index=["perturbation", "stratum"], columns="method", values="auprc")
    return wide.replace([np.inf, -np.inf], np.nan).dropna()[METHOD_ORDER]


def within_perturbation_ci(results_root: Path, n_bootstrap: int = 1000, seed: int = 42):
    """Percentile CIs from resampling candidate intervals inside each perturbation."""
    out = {}
    for path in sorted((results_root / "output").glob("*_independent_scored_candidates.tsv")):
        pert = path.name.removesuffix("_independent_scored_candidates.tsv").removeprefix("MartinRufino.")
        if pert not in PERTURBATIONS:
            continue
        scored = pd.read_csv(path, sep="\t")
        for stratum, (lo, hi) in STRATA_BOUNDS.items():
            sub = scored[(scored.distance >= lo) & (scored.distance < hi)]
            y_true = sub.positive.astype(int).to_numpy()
            if y_true.sum() < 10 or (len(y_true) - y_true.sum()) < 1:
                continue
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(y_true), size=(n_bootstrap, len(y_true)))
            for method in METHOD_ORDER:
                scores = sub[method].to_numpy()
                vals = []
                for row in idx:
                    yt, ys = y_true[row], scores[row]
                    if len(np.unique(yt)) < 2 or np.all(ys == 0):
                        continue
                    vals.append(average_precision_score(yt, ys))
                out[(pert, stratum, method)] = (
                    (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
                    if len(vals) >= 10 else (np.nan, np.nan))
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_fig4d(results_root: Path, out_dir: Path) -> None:
    """Draw mean AUPRC per distance stratum at publication dimensions."""
    # Four axes share the widths and gaps used in the publication layout.
    canvas_w, canvas_h = 165.0, 42.0
    lefts = [12.40, 51.71, 91.02, 130.33]
    width, top, height = 33.36, 4.87, 26.08

    values = eligible_table(load_auprc(results_root))
    style()
    fig = new_fig(canvas_w, canvas_h)
    axes = [add_axes_mm(fig, canvas_w, canvas_h, left, top, width, height) for left in lefts]
    ymax = 0.0
    for ax, stratum, label in zip(axes, STRATA, STRATA_LABELS):
        sub = values.xs(stratum, level="stratum")
        means, sems = sub.mean(), sub.sem()
        x = np.arange(len(METHOD_ORDER))
        ax.bar(x, means, yerr=sems, capsize=1.5,
               color=[METHOD_COLORS[m] for m in METHOD_ORDER],
               edgecolor="#777777", linewidth=0.4,
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
        ymax = max(ymax, float((means + sems).max()))
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, rotation=38, ha="right",
                           rotation_mode="anchor")
        ax.set_title(f"{label}; n={len(sub)}", pad=2.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.12)
    axes[0].set_ylabel("AUPRC\nmean\nacross perturbations", labelpad=1.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    save(fig, out_dir, "fig4d_mean_auprc")


def panel_figS8a(results_root: Path, out_dir: Path, n_bootstrap: int = 1000) -> None:
    """Per-perturbation AUPRC with within-perturbation candidate bootstrap CIs."""
    canvas_w, canvas_h = 172.0, 140.0
    left, width = 16.0, 152.0
    tops = [6.0, 37.2, 68.4, 99.6]
    height = 27.0

    values = eligible_table(load_auprc(results_root))
    cis = within_perturbation_ci(results_root, n_bootstrap=n_bootstrap)

    style()
    fig = new_fig(canvas_w, canvas_h)
    axes = [add_axes_mm(fig, canvas_w, canvas_h, left, top, width, height) for top in tops]
    bar_w = 0.15
    x = np.arange(len(PERTURBATIONS))
    for ax, stratum, label in zip(axes, STRATA, STRATA_LABELS):
        sub = values.xs(stratum, level="stratum").reindex(PERTURBATIONS)
        for i, method in enumerate(METHOD_ORDER):
            keep = sub[method].notna().to_numpy()
            if not keep.any():
                continue
            perts = list(sub.index[keep])
            heights = sub.loc[perts, method].to_numpy()
            intervals = np.array([cis.get((p, stratum, method), (np.nan, np.nan))
                                  for p in perts], dtype=float)
            err = np.vstack([heights - intervals[:, 0], intervals[:, 1] - heights])
            err = np.nan_to_num(np.clip(err, 0, None))
            ax.bar(x[keep] + (i - 2) * bar_w, heights, bar_w,
                   color=METHOD_COLORS[method], edgecolor="#777777", linewidth=0.35,
                   yerr=err, capsize=1.0,
                   error_kw={"elinewidth": 0.5, "capthick": 0.5, "ecolor": "black"},
                   label=METHOD_LABELS[i] if ax is axes[0] else None)
        ax.set_title(f"{label}; n={int(sub.notna().all(axis=1).sum())}", pad=2.0)
        ax.set_ylabel("AUPRC", labelpad=2.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(PERTURBATIONS) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([])
    ymax = float(values.max().max())
    for ax in axes:
        ax.set_ylim(0, min(1.0, ymax * 1.25))
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.12),
                   frameon=False, handlelength=1.2, handletextpad=0.5,
                   columnspacing=1.6, borderpad=0.0)
    axes[-1].set_xticklabels(PERTURBATIONS, rotation=38, ha="right", rotation_mode="anchor")
    axes[-1].set_xlabel("Perturbation", labelpad=1.5)
    save(fig, out_dir, "figS8a_per_perturbation")


def panel_figS8b(results_root: Path, out_dir: Path) -> None:
    """Paired GenPerturb-minus-comparator differences with perturbation bootstrap CIs."""
    canvas_w, canvas_h = 172.0, 40.0
    left, width, gap = 16.0, 34.0, 5.0   # axes start aligned with panel a
    top, height = 5.0, 26.0

    paired = pd.read_csv(results_root / "paired_comparison/paired_auprc_summary.tsv", sep="\t")
    paired = paired[paired.universe == "independent"]
    comparators = ["TSS_distance", "rE2G", "rE2G_extended", "ABC"]
    comp_labels = ["TSS distance", "rE2G", "rE2G ext.", "ABC"]

    style()
    fig = new_fig(canvas_w, canvas_h)
    axes = []
    for i in range(4):
        axes.append(add_axes_mm(fig, canvas_w, canvas_h,
                                left + i * (width + gap), top, width, height))
    for ax, stratum, label in zip(axes, STRATA, STRATA_LABELS):
        sub = paired[paired.stratum == stratum].set_index("comparator")
        y = np.arange(len(comparators))
        means = sub.loc[comparators, "mean_difference"].to_numpy()
        lo = means - sub.loc[comparators, "ci_low"].to_numpy()
        hi = sub.loc[comparators, "ci_high"].to_numpy() - means
        ax.axvline(0, color="#999999", linewidth=0.5)
        ax.errorbar(means, y, xerr=np.vstack([lo, hi]), fmt="o", markersize=2.2,
                    color="#1F6F6B", elinewidth=0.6, capsize=1.5, capthick=0.6,
                    linestyle="none")
        ax.set_yticks(y)
        ax.set_yticklabels(comp_labels if ax is axes[0] else [])
        ax.set_ylim(-0.6, len(comparators) - 0.4)
        n = int(sub.loc[comparators[0], "n"])
        ax.set_title(f"{label.split(' (')[0]}; n={n}", pad=2.0)
        ax.set_xlabel(r"GenPerturb $-$ comparator", labelpad=1.5)
        ax.grid(axis="x", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
    save(fig, out_dir, "figS8b_paired")


def panel_fig4f(results_root: Path, out_dir: Path, study: str = "Martin",
                canvas: tuple[float, float] = (57.24, 41.49)) -> None:
    """High-attribution motif-minus-control effects, sized for the Fig. 4f box."""
    canvas_w, canvas_h = canvas
    perts = pd.read_csv(results_root / f"{study}_tertile_effects.tsv", sep="\t")
    summary = pd.read_csv(results_root / f"{study}_tertile_effects_summary.tsv", sep="\t")
    perts = perts[perts.attr_group == "High"].copy()
    summary = summary[summary.attr_group == "High"].iloc[0]
    perts["perturbation"] = perts.source_perturbation.str.split(".").str[-1]

    # Perturbations run descending alphabetically from the top, with the
    # aggregate row above them, as in the reference panel.
    order = sorted(perts.perturbation.unique())
    y = np.arange(len(order))
    mean_y = len(order)
    indexed = perts.set_index("perturbation").loc[order]
    diff = indexed["paired_difference"].to_numpy()

    style()
    fig = new_fig(canvas_w, canvas_h)
    # Plot the paired estimand directly. The former motif/control dumbbell was
    # redundant with this difference panel; raw magnitudes remain in the source
    # table and figure legend so the scale of the contrast is still explicit.
    ax_left, ax_right_pad, ax_top, ax_bottom_pad = 9.0, 4.0, 6.0, 10.5
    ax_w = canvas_w - ax_left - ax_right_pad
    ax_h = canvas_h - ax_top - ax_bottom_pad
    ax = add_axes_mm(fig, canvas_w, canvas_h, ax_left, ax_top, ax_w, ax_h)
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.scatter(diff, y, s=3.5, color="#1F6F6B", zorder=2)
    ax.errorbar([summary.mean_paired_difference], [mean_y],
                 xerr=[[summary.mean_paired_difference - summary.ci_low],
                       [summary.ci_high - summary.mean_paired_difference]],
                 fmt="D", markersize=2.2, color="#B4485A", elinewidth=0.6,
                 capsize=1.5, capthick=0.6)
    ax.set_yticks(np.concatenate([y, [mean_y]]))
    ax.set_yticklabels(order + ["Mean"])
    ax.set_ylim(-0.7, mean_y + 0.7)
    ax.set_xlabel(r"Motif $-$ matched control: $|\Delta\Delta|$",
                  labelpad=1.0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    save(fig, out_dir, "fig4f_martin_mutation" if study == "Martin"
         else f"figS9b_{study.lower()}_mutation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auprc-root", type=Path, required=True,
                    help="Enhancer-benchmark directory containing output/ and paired_comparison/")
    ap.add_argument("--martin-tertile", type=Path, required=True,
                    help="tertile_panels/Martin_tertile directory")
    ap.add_argument("--norman-tertile", type=Path, required=True,
                    help="tertile_panels/Norman_tertile directory")
    ap.add_argument("--out", type=Path, required=True, help="Provenance record")
    ap.add_argument("--figure-out", type=Path, help="Panel SVG/PNG destination; defaults to --out")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--panels", nargs="*",
                    default=["fig4d", "figS8a", "figS8b", "fig4f", "figS9b"])
    args = ap.parse_args()

    out = output_dir(args.out)
    figure_out = args.figure_out or out
    figure_out.mkdir(parents=True, exist_ok=True)

    if "fig4d" in args.panels:
        panel_fig4d(args.auprc_root, figure_out)
    if "figS8a" in args.panels:
        panel_figS8a(args.auprc_root, figure_out, n_bootstrap=args.bootstrap)
    if "figS8b" in args.panels:
        panel_figS8b(args.auprc_root, figure_out)
    if "fig4f" in args.panels:
        panel_fig4f(args.martin_tertile, figure_out)
    if "figS9b" in args.panels:
        # Fig. S9b uses its publication dimensions rather than post-render scaling.
        panel_fig4f(args.norman_tertile, figure_out, study="Norman", canvas=(71.5, 49.2))

    provenance(out, args, [
        args.auprc_root / "output/auprc_per_perturbation.tsv",
        args.auprc_root / "paired_comparison/paired_auprc_summary.tsv",
        args.martin_tertile / "Martin_tertile_effects.tsv",
        args.martin_tertile / "Martin_tertile_effects_summary.tsv",
        args.norman_tertile / "Norman_tertile_effects.tsv",
        args.norman_tertile / "Norman_tertile_effects_summary.tsv",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
