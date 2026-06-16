#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if len(sys.argv) >= 3:
    study_name   = sys.argv[1]
    study_suffix = sys.argv[2]
else:
    study_name   = "MartinRufino2025_mixscape_exnp_train"
    study_suffix = "alphagenome_transfer_epoch100_batch256_adamw5e3"

study = f"{study_name}__{study_suffix}"
QVAL_THR = 0.05

OUT_DIR = f"figures/{study}/combined_motif"
os.makedirs(OUT_DIR, exist_ok=True)


METHOD_ORDER = [
    "GenPerturb\n(TF-MoDISco)",
    "GenPerturb\n(GimmeMotifs)",
    "rE2G extended",
    "rE2G",
    "ABC",
    "TSS ±1kbp",
    "Shuffled\nGenPerturb peaks",
]

PALETTE = {
    "Cluster match": "#4878D0",
    "Gene match":    "#EE854A",
}

GIMME_LABEL_MAP = {
    "attribution":         "GenPerturb\n(GimmeMotifs)",
    "re2g_extended":       "rE2G extended",
    "re2g":                "rE2G",
    "abc_score":           "ABC",
    "tss_1kbp":            "TSS ±1kbp",
    "attribution_shuffle": "Shuffled\nGenPerturb peaks",
}


def load_tfmodisco_stats(study, qval_thr):
    gene_path    = f"figures/{study}/tfmodisco/tfmodisco_gene_match.txt"
    cluster_path = f"figures/{study}/tfmodisco/tfmodisco_cluster_match.txt"
    rank_path    = f"figures/{study}/tfmodisco/rank_metrics_summary.txt"

    if not all(os.path.exists(p) for p in [gene_path, cluster_path, rank_path]):
        print(f"[WARN] Missing TF-MoDISco summary file(s); TF-MoDISco bar will be skipped.")
        return None

    gene_df    = pd.read_csv(gene_path, sep="\t")
    cluster_df = pd.read_csv(cluster_path, sep="\t")
    rank_df    = pd.read_csv(rank_path, sep="\t")

    gene_df    = gene_df[gene_df["qval"] < qval_thr]
    cluster_df = cluster_df[cluster_df["qval"] < qval_thr]

    matched_gene    = int(gene_df["perturbation"].nunique())
    matched_cluster = int(cluster_df["perturbation"].nunique())
    all_tf_pert     = int(rank_df["all_tf_pert"].iloc[0])

    return {
        "matched_gene": matched_gene,
        "matched_cluster": matched_cluster,
        "all_tf_pert": all_tf_pert,
        "gene_ratio":    matched_gene / all_tf_pert if all_tf_pert > 0 else 0.0,
        "cluster_ratio": matched_cluster / all_tf_pert if all_tf_pert > 0 else 0.0,
    }


def load_gimme_stats(study):
    summary_path = f"figures/{study}/gimmemotifs/motif_analysis_summary_q5e-02.txt"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"GiMMeMotifs summary not found: {summary_path}")

    df = pd.read_csv(summary_path, sep="\t")
    df["bed_type"] = df["method"].str.extract(r"\((.+)\)$")[0]
    df = df[df["bed_type"].isin(GIMME_LABEL_MAP.keys())].copy()
    return df


def build_plot_df(tfm_stats, gimme_df):
    rows = []

    if tfm_stats is not None:
        for match_type, ratio_key, count_key in [
            ("Cluster match", "cluster_ratio", "matched_cluster"),
            ("Gene match",    "gene_ratio",    "matched_gene"),
        ]:
            rows.append({
                "Method": "GenPerturb\n(TF-MoDISco)",
                "Match Type": match_type,
                "Ratio": float(tfm_stats[ratio_key]),
                "Count": int(tfm_stats[count_key]),
                "Total": int(tfm_stats["all_tf_pert"]),
            })

    for _, r in gimme_df.iterrows():
        label = GIMME_LABEL_MAP.get(r["bed_type"])
        if label is None:
            continue
        for match_type, ratio_key, count_key in [
            ("Cluster match", "cluster_ratio", "matched_cluster"),
            ("Gene match",    "gene_ratio",    "matched_gene"),
        ]:
            rows.append({
                "Method": label,
                "Match Type": match_type,
                "Ratio": float(r[ratio_key]),
                "Count": int(r[count_key]),
                "Total": int(r["all_tf_pert"]),
            })

    return pd.DataFrame(rows)


def plot_combined(plot_df, out_svg, title=f"Motif match ratio (q ≤ {QVAL_THR})"):
    method_order = [m for m in METHOD_ORDER if m in set(plot_df["Method"])]

    FONTSIZE = 23
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = FONTSIZE
    fig, ax = plt.subplots(figsize=(34/2.54, 22/2.54), dpi=300)

    sns.barplot(
        data=plot_df,
        x="Method",
        y="Ratio",
        hue="Match Type",
        palette=PALETTE,
        hue_order=["Cluster match", "Gene match"],
        order=method_order,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    if "GenPerturb\n(TF-MoDISco)" in method_order and len(method_order) > 1:
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.5)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Ratio of matched\nperturbations", fontsize=FONTSIZE)
    ax.set_xlabel("", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONTSIZE)
    ax.legend(loc="upper right", fontsize=FONTSIZE, title=None, frameon=True)

    plt.tight_layout()
    plt.savefig(out_svg)
    plt.close()


if __name__ == "__main__":
    print(f"[INFO] study = {study}")

    tfm_stats = load_tfmodisco_stats(study, QVAL_THR)
    if tfm_stats is not None:
        print(f"[INFO] TF-MoDISco (q < {QVAL_THR}): "
              f"gene={tfm_stats['matched_gene']}/{tfm_stats['all_tf_pert']} "
              f"({tfm_stats['gene_ratio']:.2f}), "
              f"cluster={tfm_stats['matched_cluster']}/{tfm_stats['all_tf_pert']} "
              f"({tfm_stats['cluster_ratio']:.2f})")

    gimme_df = load_gimme_stats(study)
    print(f"[INFO] GiMMeMotifs (BH-FDR q <= {QVAL_THR}): loaded "
          f"{len(gimme_df)} rows for bed_types {sorted(gimme_df['bed_type'].unique())}")

    plot_df = build_plot_df(tfm_stats, gimme_df)
    if plot_df.empty:
        print("[ERROR] No data to plot. Aborting.")
        sys.exit(1)

    tsv_out = f"{OUT_DIR}/matched_genes_ratio_combined_q005.tsv"
    plot_df.to_csv(tsv_out, sep="\t", index=False)
    print(f"[INFO] Saved data: {tsv_out}")

    svg_out = f"{OUT_DIR}/matched_genes_ratio_combined_q005.svg"
    plot_combined(plot_df, svg_out)
    print(f"[INFO] Saved plot: {svg_out}")

    print("Done!")
