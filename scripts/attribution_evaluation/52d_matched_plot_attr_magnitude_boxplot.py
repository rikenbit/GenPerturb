#!/usr/bin/env python
import argparse
import importlib.util
import os
from pathlib import Path
import warnings
from glob import glob

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
os.chdir(CWD)

_CFG_PATH = os.path.join(CWD, "scripts/attribution_evaluation/52_study_config.py")
_spec = importlib.util.spec_from_file_location("study_config_52", _CFG_PATH)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
STUDY_CONFIGS = _cfg.STUDY_CONFIGS
DATA_DIR_TPL = _cfg.DATA_DIR_TPL
FIG_DIR_TPL = _cfg.FIG_DIR_TPL
TABLE_DIR_TPL = _cfg.TABLE_DIR_TPL

GROUP_LABELS = ["High attr", "Mid attr", "Low attr", "Neg ctrl"]
GROUP_COLORS = ["#2166AC", "#67A9CF", "#D1E5F0", "#F4A582"]
Y_LABEL = "Perturbation-effect cancellation (ΔΔPred)"

OUTPUT_BASENAME = "attr_magnitude_grouped_source_only_boxplot_matched"

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def parse_args():
    matched_keys = [k for k in STUDY_CONFIGS.keys() if k.endswith("_matched")]
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=matched_keys + ["all"])
    return p.parse_args()


def load_attr_lookup(data_dir):
    attr = pd.read_csv(f"{data_dir}/seqlet_attribution.tsv", sep="\t")
    if len(attr) == 0:
        return {}
    return {
        (row["site_id"], row["perturbation"]): float(row["attr_sum_abs"])
        for _, row in attr.iterrows()
        if pd.notna(row["attr_sum_abs"])
    }


def collect_arrays(data_dir):
    attr_lookup = load_attr_lookup(data_dir)

    h5_files = sorted(glob(f"{data_dir}/results/*_predictions.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No prediction H5 under {data_dir}/results/")

    source_attr = []
    source_absD = []
    negctrl_absD = []

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                wt_pred = np.array(f["wt_pred"])
                mt_pred = np.array(f["mt_pred_mean"])
                expr_cols = [s.decode() if isinstance(s, bytes) else s
                             for s in f["metadata"]["expr_cols"][:]]
                site_ids = [s.decode() if isinstance(s, bytes) else s
                            for s in f["metadata"]["site_ids"][:]]
                mutation_types = [s.decode() if isinstance(s, bytes) else s
                                  for s in f["metadata"]["mutation_types"][:]]
                ctrl_raw = f["metadata"]["control_col"][()]
                ctrl_col = ctrl_raw.decode() if isinstance(ctrl_raw, bytes) else str(ctrl_raw)
        except Exception as e:
            print(f"  Skip {os.path.basename(h5_path)}: {e}")
            continue

        ctrl_idx = expr_cols.index(ctrl_col)
        pert_indices = [i for i, c in enumerate(expr_cols) if c != ctrl_col]
        pert_cols = [expr_cols[i] for i in pert_indices]

        wt_ctrl = float(wt_pred[ctrl_idx])
        wt_fc = wt_pred[pert_indices] - wt_ctrl

        for ti in range(len(site_ids)):
            mt_row = mt_pred[ti]
            if np.array_equal(mt_row, wt_pred):
                continue
            mt_ctrl = float(mt_row[ctrl_idx])
            mt_fc = mt_row[pert_indices] - mt_ctrl
            abs_delta = np.abs(mt_fc - wt_fc)

            mtype = mutation_types[ti]
            if mtype == "neg_control":
                negctrl_absD.extend(abs_delta.tolist())
            elif mtype == "seqlet_mut":
                site_id = site_ids[ti]
                for pi, pc in enumerate(pert_cols):
                    a = attr_lookup.get((site_id, pc))
                    if a is None:
                        continue
                    source_attr.append(a)
                    source_absD.append(float(abs_delta[pi]))

    return (np.asarray(source_attr, dtype=float),
            np.asarray(source_absD, dtype=float),
            np.asarray(negctrl_absD, dtype=float))


def build_box_data(source_attr, source_absD, negctrl_absD):
    if len(source_attr) < 20 or len(negctrl_absD) == 0:
        raise RuntimeError(
            f"Insufficient data: source_seqlet={len(source_attr)}, "
            f"neg_control={len(negctrl_absD)}")

    tertiles = pd.qcut(source_attr, q=3, labels=["Low", "Mid", "High"])
    box_data = [
        source_absD[tertiles == "High"],
        source_absD[tertiles == "Mid"],
        source_absD[tertiles == "Low"],
        negctrl_absD,
    ]
    n_per_group = dict(zip(GROUP_LABELS, [len(g) for g in box_data]))
    return box_data, n_per_group


def format_pvalue(p):
    if p == 0 or not np.isfinite(p):
        return "p < 1e-300"
    return f"p = {p:.2e}"


def plot_one_study(study, figures_dir, tables_dir):
    data_dir = DATA_DIR_TPL.format(study=study)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    print(f"[{study}] Reading {data_dir}/results/*_predictions.h5")
    source_attr, source_absD, negctrl_absD = collect_arrays(data_dir)
    box_data, n_per_group = build_box_data(source_attr, source_absD, negctrl_absD)

    mw_stat, mw_pval = stats.mannwhitneyu(
        box_data[0], box_data[3], alternative="greater")

    fig, ax = plt.subplots(figsize=(9, 7))
    bp = ax.boxplot(box_data, labels=GROUP_LABELS,
                    showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], GROUP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("orange")
        median.set_linewidth(1.5)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"{study}")

    ax.text(0.95, 0.95, format_pvalue(mw_pval),
            transform=ax.transAxes, fontsize=18, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="lightgray", alpha=0.85))

    plt.tight_layout()
    svg_path = f"{figures_dir}/{OUTPUT_BASENAME}.svg"
    png_path = f"{figures_dir}/{OUTPUT_BASENAME}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {svg_path}")
    print(f"  Saved: {png_path}")

    counts_df = pd.DataFrame({
        "study": [study] * 4,
        "group": GROUP_LABELS,
        "n": [n_per_group[g] for g in GROUP_LABELS],
        "median_abs_delta_fc": [float(np.median(g)) if len(g) else float("nan")
                                for g in box_data],
        "high_vs_negctrl_mw_pvalue": [mw_pval] * 4,
    })
    counts_path = f"{tables_dir}/attr_magnitude_group_counts_matched.tsv"
    counts_df.to_csv(counts_path, sep="\t", index=False)
    print(f"  Saved: {counts_path}")
    print(f"  n per group: {n_per_group}")
    print(f"  High vs NegCtrl MW p-value: {mw_pval:.3e}")


def main():
    args = parse_args()
    matched_keys = [k for k in STUDY_CONFIGS.keys() if k.endswith("_matched")]
    studies = matched_keys if args.study == "all" else [args.study]

    for study in studies:
        cfg = STUDY_CONFIGS[study]
        figures_dir = FIG_DIR_TPL.format(study_full=cfg["study_full"])
        tables_dir = TABLE_DIR_TPL.format(study=study)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        plot_one_study(study, figures_dir, tables_dir)


if __name__ == "__main__":
    main()
