#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODEL_TAG = "alphagenome_transfer_epoch100_batch256_adamw5e3"
STUDY = "Martin"
STUDY_TAG = "MartinRufino2025_mixscape_exnp_train"
STUDY_FULL = f"{STUDY_TAG}__{MODEL_TAG}"
WORK_DIR = REPO_ROOT / "figures" / STUDY_FULL / "selected_martin_attribution_figures"
MPLCONFIG_DIR = WORK_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


AXIS_DIR = WORK_DIR / "axis_validation" / STUDY / "tables"
PER_PERT_DIR = AXIS_DIR / "per_pert"
AGG_DIR = WORK_DIR / "aggregation" / STUDY / "tables"
RANK_DIR = WORK_DIR / "diff_residual" / STUDY / "tables"
WITHIN_DIR = WORK_DIR / "within_gene" / STUDY / "tables"
FIGURE_DIR = WORK_DIR / "figures"

AXES = {"Ctrl": "sum_abs_ctrl", "Pert": "sum_abs_pert", "Diff": "sum_abs_diff"}
AXIS_ORDER = ["Ctrl", "Pert", "Diff"]
AXIS_LABEL = {"Ctrl": "Σ|control|", "Pert": "Σ|perturbation|", "Diff": "Σ|differential|"}
AXIS_COLOR = {"Ctrl": "#7DA8C7", "Pert": "#B97AAA", "Diff": "#DD8452"}

K_GRID = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
TARGET_COL = "obs_abs_logFC"
TARGET_LABEL = "observed |log fold-change|"
TARGET_SLUG = "observed_abslog_fold_changeabs"

FC_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
FC_BIN_LABELS = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]
FC_METHODS = ["raw_pert", "diff"]
FC_SCORE_COL = {"raw_pert": "sum_abs_pert", "diff": "sum_abs_diff"}
FC_LABEL = {
    "raw_pert": "Perturbation attribution  (no control subtraction)",
    "diff": "Differential attribution",
}
FC_COLOR = {"raw_pert": "#4C78A8", "diff": "#F58518"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_per_pert() -> list[Path]:
    tsvs = sorted(PER_PERT_DIR.glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No local per-pert TSVs found under {PER_PERT_DIR}")
    return tsvs


def merge_per_pert(tsvs: list[Path]) -> pd.DataFrame:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    merged = pd.concat((pd.read_csv(tsv, sep="\t") for tsv in tsvs), ignore_index=True)
    out = AGG_DIR / "merged.tsv.gz"
    merged.to_csv(out, sep="\t", index=False, compression="gzip")
    print(f"[saved] {display_path(out)} ({merged['perturbation'].nunique()} perts, {len(merged):,} rows)")
    return merged


def compute_ksweep(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cols = [TARGET_COL, *AXES.values()]
    for pert, group in merged.groupby("perturbation", sort=False):
        df = group[cols].dropna(subset=cols)
        df = df[np.isfinite(df[cols]).all(axis=1)].reset_index(drop=True)
        if len(df) < 100:
            continue
        target = df[TARGET_COL].to_numpy(dtype=float)
        target_topk = {}
        for k_frac in K_GRID:
            k = max(1, int(round(len(df) * k_frac)))
            target_topk[k_frac] = set(np.argpartition(target, -k)[-k:].tolist())

        for axis in AXIS_ORDER:
            values = df[AXES[axis]].to_numpy(dtype=float)
            for k_frac in K_GRID:
                k = max(1, int(round(len(df) * k_frac)))
                axis_top = set(np.argpartition(values, -k)[-k:].tolist())
                target_top = target_topk[k_frac]
                overlap = len(axis_top & target_top)
                expected = (k * k) / len(df)
                rows.append(
                    {
                        "pert": pert,
                        "axis": axis,
                        "k_frac": k_frac,
                        "k": k,
                        "n_genes": len(df),
                        "overlap": overlap,
                        "fold_enrichment": overlap / expected if expected > 0 else np.nan,
                        "jaccard": overlap / len(axis_top | target_top) if axis_top or target_top else np.nan,
                    }
                )

    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["axis", "k_frac"])
        .agg(
            n_perts=("pert", "nunique"),
            mean_fe=("fold_enrichment", "mean"),
            se_fe=("fold_enrichment", lambda s: float(s.std(ddof=1) / np.sqrt(s.count()))),
            median_fe=("fold_enrichment", "median"),
            mean_jaccard=("jaccard", "mean"),
        )
        .reset_index()
    )
    RANK_DIR.mkdir(parents=True, exist_ok=True)
    long_out = RANK_DIR / f"ksweep_long_{TARGET_SLUG}.tsv"
    summary_out = RANK_DIR / f"ksweep_summary_{TARGET_SLUG}.tsv"
    long_df.to_csv(long_out, sep="\t", index=False)
    summary.to_csv(summary_out, sep="\t", index=False)
    print(f"[saved] {display_path(summary_out)}")
    return long_df, summary


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def fc_bin_label(max_fc: float) -> str:
    for lo, hi, label in zip(FC_BIN_EDGES[:-1], FC_BIN_EDGES[1:], FC_BIN_LABELS):
        if max_fc >= lo and max_fc < hi:
            return label
    return FC_BIN_LABELS[-1]


def compute_fc_binned_gene_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, sub in merged.groupby("gene", sort=False):
        y = sub[TARGET_COL].to_numpy(dtype=float)
        max_fc = float(np.max(y))
        for method in FC_METHODS:
            x = sub[FC_SCORE_COL[method]].to_numpy(dtype=float)
            rho = safe_spearman(x, y)
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "gene": gene,
                    "target_kind": "observed",
                    "method": method,
                    "fc_bin": fc_bin_label(max_fc),
                    "max_fc": max_fc,
                    "target_range": float(np.max(y) - np.min(y)),
                    "n_perts": len(sub),
                    "spearman": rho,
                }
            )
    per_gene = pd.DataFrame(rows)
    WITHIN_DIR.mkdir(parents=True, exist_ok=True)
    out = WITHIN_DIR / "fc_binned_gene_wise_correlations_observed.tsv"
    per_gene.to_csv(out, sep="\t", index=False)
    print(f"[saved] {display_path(out)} ({len(per_gene):,} rows)")
    return per_gene


def plot_rank_ksweep(ksweep_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for axis in AXIS_ORDER:
        sub = ksweep_summary[ksweep_summary["axis"] == axis].sort_values("k_frac")
        xpct = sub["k_frac"].to_numpy(dtype=float) * 100
        mean = sub["mean_fe"].to_numpy(dtype=float)
        se = sub["se_fe"].to_numpy(dtype=float)
        ax.plot(xpct, mean, "-o", ms=3.5, lw=1.6, color=AXIS_COLOR[axis], label=AXIS_LABEL[axis])
        ax.fill_between(xpct, mean - se, mean + se, color=AXIS_COLOR[axis], alpha=0.18, lw=0)

    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 25])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "25"], fontsize=8)
    ax.set_xlabel("Shared top-gene cutoff applied separately\n" "to |FC| and Σ|attribution| rankings (%)")
    ax.set_ylabel("Overlap enrichment between the cutoff-selected\n" "|FC| and Σ|attribution| gene sets")
    ax.set_title("Martin — Top-k enrichment vs cut size")
    ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"rank_ksweep_{TARGET_SLUG}.svg"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fc_binned(per_gene: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    width = 0.32
    base = np.arange(len(FC_BIN_LABELS))

    for i, method in enumerate(FC_METHODS):
        data = [
            per_gene[(per_gene["method"] == method) & (per_gene["fc_bin"] == bin_label)]["spearman"]
            .dropna()
            .to_numpy(dtype=float)
            for bin_label in FC_BIN_LABELS
        ]
        positions = base + (i - 0.5) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(FC_COLOR[method])
            patch.set_alpha(0.70)
            patch.set_edgecolor("#333333")
        for whisker in bp["whiskers"]:
            whisker.set_color("#333333")
        for cap in bp["caps"]:
            cap.set_color("#333333")

    counts = [int(per_gene[per_gene["fc_bin"] == bin_label]["gene"].nunique()) for bin_label in FC_BIN_LABELS]
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(base)
    ax.set_xticklabels([f"{bin_label}\n(n={count})" for bin_label, count in zip(FC_BIN_LABELS, counts)])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Gene-level maximum observed |FC| across perturbations")
    ax.set_ylabel("Gene-wise Spearman rho")
    ax.set_title("Martin: attribution-|FC| correlation stratified by response magnitude", pad=14)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=FC_COLOR[method], lw=8, alpha=0.70, label=FC_LABEL[method])
        for method in FC_METHODS
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        fontsize=9,
        handlelength=2.4,
        columnspacing=2.6,
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.09, right=0.98)

    out = FIGURE_DIR / "fc_binned_gene_wise_correlation_observed.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"[work-dir] {display_path(WORK_DIR)}")
    tsvs = require_per_pert()
    merged = merge_per_pert(tsvs)
    _, ksweep_summary = compute_ksweep(merged)
    fc_binned = compute_fc_binned_gene_correlations(merged)
    outputs = [plot_rank_ksweep(ksweep_summary), plot_fc_binned(fc_binned)]
    for out in outputs:
        print(f"[saved] {display_path(out)}")


if __name__ == "__main__":
    main()
