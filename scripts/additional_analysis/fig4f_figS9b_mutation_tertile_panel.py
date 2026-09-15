#!/usr/bin/env python
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import read_tsv, output_dir, provenance, save, bootstrap_mean

MOTIF_COLOR = "#267d78"
CONTROL_COLOR = "#9aa0a6"
MEAN_COLOR = "#bc5265"
LINE_COLOR = "#c9c9c9"


def mutation_module():
    """Load the aggregation used by the pipeline so it is defined in one place."""
    path = Path(__file__).resolve().parent / "fig4f_figS9b_mutation_aggregation.py"
    spec = importlib.util.spec_from_file_location("mutation_aggregation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style(font_pt):
    # Sizes are final sizes: the panel is composed at 1:1, so no down-scaling.
    # The default matches the neighbouring panels of Fig. 4 and Fig. S9, whose
    # composed text measures about 3.0-4.3 pt.
    plt.rcParams.update({
        "font.size": font_pt, "axes.labelsize": font_pt, "axes.titlesize": font_pt,
        "xtick.labelsize": font_pt - .5, "ytick.labelsize": font_pt - .5,
        "legend.fontsize": font_pt - .8, "axes.linewidth": .35,
        "xtick.major.width": .35, "ytick.major.width": .35,
        "xtick.major.size": 1.2, "ytick.major.size": 1.2,
        "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
        "font.family": "sans-serif", "svg.fonttype": "none",
    })


def draw(pert, summary, tertiles, width_mm, height_mm, font_pt):
    style(font_pt)
    # Marker and line weights follow the text size so the panel keeps the same
    # visual weight as the rest of the figure; line widths keep a print floor.
    k = font_pt / 7.0
    ms, lw = 2.2 * k, max(.5, .8 * k)
    fig, (axm, axd) = plt.subplots(
        1, 2, figsize=(width_mm / 25.4, height_mm / 25.4), layout="constrained",
        gridspec_kw=dict(width_ratios=[1, 1]))
    fig.get_layout_engine().set(w_pad=.006, h_pad=.006, wspace=.02, hspace=0)
    x = np.arange(len(tertiles))
    wide = {column: pert.pivot(index="source_perturbation", columns="attr_group",
                               values=column)[tertiles]
            for column in ["paired_difference", "abs_delta_delta", "abs_delta_delta_control"]}

    # Left: the two magnitudes the difference is taken on. Plotting both makes the
    # control flatness across tertiles visible, which is the specificity claim.
    means = summary.set_index("attr_group").loc[tertiles]
    for prefix, color, label in [("control", CONTROL_COLOR, "Matched control"),
                                 ("motif", MOTIF_COLOR, "Motif")]:
        centre = means[f"mean_{prefix}"].to_numpy()
        err = np.vstack([centre - means[f"{prefix}_ci_low"], means[f"{prefix}_ci_high"] - centre])
        axm.errorbar(x, centre, yerr=err, fmt="o-", color=color, ms=ms, lw=lw,
                     elinewidth=max(.4, .6 * k), capsize=1.4 * k, label=label)
    axm.set_ylabel("|ΔΔ| (fitted units)", labelpad=1.2)
    axm.set_ylim(bottom=0)
    axm.legend(frameon=False, handletextpad=.3, borderpad=.1, labelspacing=.2,
               loc="upper left", handlelength=1.1, borderaxespad=.1)

    # Right: the paired quantity the summary and its bootstrap interval come from.
    # One faint line per perturbation shows that the trend is not driven by a few.
    for _, row in wide["paired_difference"].iterrows():
        axd.plot(x, row.to_numpy(), "-", color=LINE_COLOR, lw=max(.3, .4 * k), zorder=1)
    axd.scatter(np.tile(x, len(wide["paired_difference"])),
                wide["paired_difference"].to_numpy().ravel(),
                s=3 * k, color=MOTIF_COLOR, zorder=2, linewidths=0)
    axd.errorbar(x, means.mean_paired_difference,
                 yerr=np.vstack([means.mean_paired_difference - means.ci_low,
                                 means.ci_high - means.mean_paired_difference]),
                 fmt="D", color=MEAN_COLOR, ms=2.6 * k, lw=0, elinewidth=max(.4, .7 * k),
                 capsize=1.6 * k, zorder=3, label="Mean (95% CI)")
    axd.axhline(0, color="gray", lw=max(.3, .4 * k))
    axd.set_ylabel("Motif − control |ΔΔ|", labelpad=1.2)
    axd.legend(frameon=False, handletextpad=.3, borderpad=.1, loc="upper left",
               handlelength=1.0, borderaxespad=.1)

    for ax in (axm, axd):
        # Headroom so the in-axes legend clears the topmost tick label and data.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + .18 * (high - low))
        ax.set_xticks(x, tertiles)
        ax.set_xlim(-.35, len(tertiles) - .65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Differential-attribution tertile", fontsize=font_pt)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="mutation_pairs.tsv from fig4f_figS9b_mutation_aggregation.py")
    p.add_argument("--label", required=True, help="Short study label used in output names")
    p.add_argument("--out", required=True, help="Source-data tables and run.json")
    p.add_argument("--figure-out", help="Panel SVG/PNG destination; defaults to --out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--width-mm", type=float, default=78.0, help="Destination box width in the composite")
    p.add_argument("--height-mm", type=float, default=42.0, help="Destination box height in the composite")
    p.add_argument("--font-pt", type=float, default=4.3,
                   help="Final composed text size; matches the neighbouring panels of Fig. 4 and S9")
    a = p.parse_args()
    module = mutation_module()
    columns = ["gene", "source_perturbation", "attr_group", "paired_complete"] + module.AGGREGATED
    pairs = read_tsv(a.pairs, columns)
    pairs = pairs[pairs.paired_complete]
    if not pairs[module.AGGREGATED].notna().all().all():
        raise ValueError(f"{a.pairs}: complete pairs carry missing effect values")
    missing = set(module.TERTILES) - set(pairs.attr_group)
    if missing:
        raise ValueError(f"{a.pairs}: missing attribution tertiles {sorted(missing)}")
    gene, pert = module.aggregate_tertiles(pairs)
    counts = pert.groupby("attr_group").source_perturbation.nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Tertiles cover different perturbation sets: {counts.to_dict()}")
    rows = []
    for group in module.TERTILES:
        block = pert[pert.attr_group == group]
        row = dict(attr_group=group,
                   **module.summarise(block.paired_difference.to_numpy(), a.seed, a.bootstrap))
        # The displayed magnitudes carry the same perturbation-bootstrap interval as
        # the paired difference, so no interval on this panel is a within-site one.
        for prefix, column in [("motif", "abs_delta_delta"), ("control", "abs_delta_delta_control")]:
            values = block[column].to_numpy()
            row[f"mean_{prefix}"] = values.mean()
            row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = bootstrap_mean(values, a.seed, a.bootstrap)
        row["magnitude_ratio"] = row["mean_motif"] / row["mean_control"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = output_dir(a.out)
    figure_out = out
    if a.figure_out:
        figure_out = Path(a.figure_out)
        figure_out.mkdir(parents=True, exist_ok=True)
    fig = draw(pert, summary, module.TERTILES, a.width_mm, a.height_mm, a.font_pt)
    name = f"{a.label}_tertile_effects"
    fig.savefig(figure_out / f"{name}.svg")
    fig.savefig(figure_out / f"{name}.png", dpi=600)
    plt.close(fig)
    save(pert, out, f"{name}.tsv")
    save(gene, out, f"{name}_gene_level.tsv")
    save(summary, out, f"{name}_summary.tsv")
    provenance(out, a, [a.pairs])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
