#!/usr/bin/env python3

import os
import sys
from typing import Optional, List, Tuple, Dict

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

gimme_roc_pval_threshold: Optional[float] = 1e-4
gimme_bed_types_roc: List[str] = [
    "attribution_re2g_extended", "attribution_re2g_extended_shuffle",
    "attribution_re2g", "attribution_re2g_shuffle",
    "attribution_abc", "attribution_abc_shuffle",
    "attribution", "attribution_shuffle",
    "re2g_extended", "re2g_extended_shuffle",
    "re2g", "re2g_shuffle",
    "abc_score", "abc_score_shuffle",
    "tss_1kbp"
]

DEBUG = True

os.makedirs(f"figures/{study}/gimmemotifs", exist_ok=True)

tf_list = pd.read_csv(
    "reference/humantfs/DatabaseExtract_v_1.01.txt",
    sep="\t",
    usecols=["HGNC symbol"]
)["HGNC symbol"].to_list()

tfs = [i for i in os.listdir(f"attribution/{study}/") if any(j in i for j in tf_list)]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    valid_mask = ~np.isnan(pvals)
    valid = pvals[valid_mask]
    m = len(valid)
    qvals = np.full(n, np.nan, dtype=float)
    if m == 0:
        return qvals
    order = np.argsort(valid)
    ranked = valid[order]
    ranks = np.arange(1, m + 1)
    q_ranked = ranked * m / ranks
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    qvals[valid_mask] = q_valid
    return qvals


def load_cluster_info() -> pd.DataFrame:
    cluster = pd.read_csv("reference/jaspar/clusters.tab", sep="\t", usecols=[0, 2])
    cluster = (
        cluster.set_index("cluster")["name"]
        .str.split(",", expand=True)
        .stack()
        .str.upper()
        .reset_index()
        .rename(columns={0: "cluster_gene"})
        .drop("level_1", axis=1)
    )
    cluster = (
        cluster.set_index("cluster")["cluster_gene"]
        .str.split("::", expand=True)
        .stack()
        .str.strip()
        .reset_index()
        .rename(columns={0: "cluster_gene"})
        .drop("level_1", axis=1)
    )
    cluster = cluster.drop_duplicates()
    return cluster


def expand_genes(df: pd.DataFrame, col_name: str, new_col_name: str, separator: str, index_col: str) -> pd.Series:
    s = df.set_index(index_col)[col_name]
    s = s.fillna("").astype(str)
    split_genes = s.str.split(separator, expand=True)
    split_genes = split_genes.stack().reset_index(level=1, drop=True).str.upper()
    split_genes.name = new_col_name
    return split_genes


def parse_gimme_motif_to_id_and_gene(motif: str) -> Tuple[str, str]:
    s = str(motif).strip()
    if "_" in s:
        motif_id, rest = s.split("_", 1)
    else:
        motif_id, rest = s, s

    if "." in rest:
        gene_part = rest.rsplit(".", 1)[-1]
    else:
        gene_part = rest

    return motif_id.strip(), gene_part.strip().upper()


def load_gimme_roc_report_for_one(
    roc_path: str,
    pert: str,
    bed_type: str,
    pval_threshold: Optional[float],
    top_n: Optional[int],
    qval_threshold: Optional[float] = None,
) -> pd.DataFrame:
    if DEBUG:
        print(f"  [DEBUG] Attempting to load: {roc_path}")
        print(f"  [DEBUG] File exists: {os.path.exists(roc_path)}")

    df = pd.read_csv(roc_path, sep="\t", engine="python")

    if DEBUG:
        print(f"  [DEBUG] Loaded {len(df)} rows")

    if "Motif" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Motif"})
    df["Motif"] = df["Motif"].fillna("").astype(str)

    pcol = None
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc in ("p-value", "pvalue", "p_value"):
            pcol = c
            break
    if pcol is None:
        for c in df.columns:
            if "p-value" in str(c).strip().lower():
                pcol = c
                break
    if pcol is None:
        raise ValueError(f"[ERROR] P-value column not found in {roc_path}. Columns={list(df.columns)}")

    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=[pcol]).copy()

    df = df.sort_values(by=pcol, ascending=True).reset_index(drop=True)
    df["motif_rank"] = df.index + 1  # 1-indexed rank in full p-value sorted list

    df["qvalue_bh"] = benjamini_hochberg(df[pcol].to_numpy())

    if qval_threshold is not None:
        df_filtered = df[df["qvalue_bh"] <= float(qval_threshold)]
        if top_n is not None:
            df_pick = df_filtered.head(int(top_n)).copy()
            pick_mode = f"q<={qval_threshold}_n{len(df_pick)}" if len(df_pick) > 0 else f"q<={qval_threshold}_none"
        else:
            df_pick = df_filtered.copy()
            pick_mode = f"q<={qval_threshold}_all_n{len(df_pick)}" if len(df_pick) > 0 else f"q<={qval_threshold}_none"
    elif pval_threshold is not None:
        df_filtered = df[df[pcol] <= float(pval_threshold)]
        if top_n is not None:
            df_pick = df_filtered.head(int(top_n)).copy()
            pick_mode = f"p<={pval_threshold}_n{len(df_pick)}" if len(df_pick) > 0 else f"p<={pval_threshold}_none"
        else:
            df_pick = df_filtered.copy()
            pick_mode = f"p<={pval_threshold}_all_n{len(df_pick)}" if len(df_pick) > 0 else f"p<={pval_threshold}_none"
    else:
        if top_n is not None:
            df_pick = df.head(int(top_n)).copy()
            pick_mode = f"top{top_n}"
        else:
            df_pick = df.copy()
            pick_mode = f"all_n{len(df)}"

    if DEBUG:
        print(f"  [DEBUG] After filtering: {len(df_pick)} motifs (pick_mode={pick_mode})")
        if len(df_pick) > 0:
            print(f"  [DEBUG] P-value range: {df_pick[pcol].min():.2e} to {df_pick[pcol].max():.2e}")
            print(f"  [DEBUG] Sample motifs: {df_pick['Motif'].head(3).tolist()}")
        else:
            print("  [DEBUG] No motifs passed the threshold")

    if df_pick.empty:
        return pd.DataFrame(columns=[
            "Motif", "perturbation", "pert_gene", "bed_type", "pick_mode",
            "source_file", "motif_id", "motif_gene", "motif_rank"
        ])

    df_pick["perturbation"] = pert
    df_pick["pert_gene_raw"] = pert.split(".")[1] if "." in pert else pert
    df_pick["bed_type"] = bed_type
    df_pick["pick_mode"] = pick_mode
    df_pick["source_file"] = roc_path  # for debugging

    tmp = df_pick["Motif"].apply(parse_gimme_motif_to_id_and_gene)
    df_pick["motif_id"] = tmp.apply(lambda x: x[0])
    df_pick["motif_gene_raw"] = tmp.apply(lambda x: x[1])

    expanded_motif = expand_genes(
        df_pick.assign(motif_gene=df_pick["motif_gene_raw"]),
        "motif_gene",
        "motif_gene",
        "::",
        "Motif"
    )
    df_pick = pd.merge(
        df_pick.drop("motif_gene_raw", axis=1),
        expanded_motif.reset_index().drop_duplicates(),
        on="Motif",
        how="left"
    )

    expanded_pert = expand_genes(
        df_pick.assign(pert_gene=df_pick["pert_gene_raw"]),
        "pert_gene",
        "pert_gene",
        "_",
        "perturbation"
    )
    df_pick = pd.merge(
        df_pick.drop(["pert_gene_raw"], axis=1),
        expanded_pert.reset_index().drop_duplicates(),
        on="perturbation",
        how="left"
    )

    df_pick = df_pick.drop_duplicates()
    return df_pick


def load_gimme_roc_reports(
    study: str,
    tfs: List[str],
    bed_types: List[str],
    pval_threshold: Optional[float],
    top_n: Optional[int],
    qval_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    all_results: List[pd.DataFrame] = []
    bed_type_pert_counts: Dict[str, int] = {}

    if DEBUG:
        print("\n[DEBUG] ===== File existence check =====")
        for pert in tfs[:3]:
            for bed_type in bed_types:
                roc_path = f"gimme_results/{study}/{pert}/{bed_type}/gimme.roc.report.txt"
                print(f"[DEBUG] {bed_type:25s} {pert:20s} -> {os.path.exists(roc_path)}")
        print("[DEBUG] =====================================\n")

    for pert in tfs:
        for bed_type in bed_types:
            roc_path = f"gimme_results/{study}/{pert}/{bed_type}/gimme.roc.report.txt"
            if not os.path.exists(roc_path):
                if DEBUG:
                    print(f"  [DEBUG] SKIPPED (not exists): {roc_path}")
                continue

            bed_type_pert_counts[bed_type] = bed_type_pert_counts.get(bed_type, 0) + 1

            try:
                if DEBUG:
                    print(f"\n[DEBUG] Processing: pert={pert}, bed_type={bed_type}")
                one = load_gimme_roc_report_for_one(
                    roc_path=roc_path,
                    pert=pert,
                    bed_type=bed_type,
                    pval_threshold=pval_threshold,
                    top_n=top_n,
                    qval_threshold=qval_threshold,
                )
                if not one.empty:
                    all_results.append(one)
            except Exception as e:
                print(f"  [WARN] Failed to parse {roc_path}: {e}")
                continue

    if not all_results:
        return pd.DataFrame(), bed_type_pert_counts
    return pd.concat(all_results, ignore_index=True), bed_type_pert_counts


def process_gimme_roc_with_cluster(df: pd.DataFrame, cluster: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cluster = cluster.copy()
    cluster["cluster_gene"] = cluster["cluster_gene"].astype(str).str.upper()

    out = df.copy()
    out["motif_gene"] = out["motif_gene"].astype(str).str.upper()
    out["pert_gene"] = out["pert_gene"].astype(str).str.upper()

    cluster_df = pd.merge(
        out,
        cluster.rename(columns={"cluster": "motif_cluster"}),
        left_on="motif_gene",
        right_on="cluster_gene",
        how="left"
    )
    cluster_df = pd.merge(
        cluster_df,
        cluster.rename(columns={"cluster": "pert_cluster"}),
        left_on="pert_gene",
        right_on="cluster_gene",
        how="left",
        suffixes=("", "_pert")
    )
    cluster_df = cluster_df.drop_duplicates()

    cluster_df["gene_match_flag"] = (cluster_df["motif_gene"] == cluster_df["pert_gene"]).astype(int)
    cluster_df["cluster_match_flag"] = (cluster_df["motif_cluster"] == cluster_df["pert_cluster"]).astype(int)

    return cluster_df


def calculate_stats(cluster_df: pd.DataFrame, method_name: str, threshold_type: str, all_tf_pert: int) -> Dict[str, object]:
    if cluster_df is None or cluster_df.empty:
        return {
            "method": method_name,
            "threshold": threshold_type,
            "matched_gene": 0,
            "matched_cluster": 0,
            "all_tf_pert": int(all_tf_pert)
        }

    num_match_gene = len(set(cluster_df.query("gene_match_flag == 1")["perturbation"]))
    num_match_clst = len(set(cluster_df.query("cluster_match_flag == 1")["perturbation"]))

    return {
        "method": method_name,
        "threshold": threshold_type,
        "matched_gene": int(num_match_gene),
        "matched_cluster": int(num_match_clst),
        "all_tf_pert": int(all_tf_pert)
    }


def calculate_rank_stats(
    cluster_df: pd.DataFrame,
    method_name: str,
    all_tf_pert: int,
    tfs_with_file: List[str]
) -> Dict[str, object]:
    empty_result = {
        "method": method_name,
        "mrr_gene": 0.0, "mrr_cluster": 0.0,
        "top1_gene": 0.0, "top1_cluster": 0.0,
        "top5_gene": 0.0, "top5_cluster": 0.0,
        "top10_gene": 0.0, "top10_cluster": 0.0,
        "median_rank_gene": np.nan, "median_rank_cluster": np.nan,
        "all_tf_pert": int(all_tf_pert),
    }
    if cluster_df is None or cluster_df.empty or not tfs_with_file:
        return empty_result

    best_ranks = []
    for pert in tfs_with_file:
        sub = cluster_df[cluster_df["perturbation"] == pert]
        if sub.empty:
            best_ranks.append({"best_gene_rank": np.inf, "best_cluster_rank": np.inf})
            continue
        gene_matches = sub[sub["gene_match_flag"] == 1]
        cluster_matches = sub[sub["cluster_match_flag"] == 1]
        best_gene_rank = gene_matches["motif_rank"].min() if not gene_matches.empty else np.inf
        best_cluster_rank = cluster_matches["motif_rank"].min() if not cluster_matches.empty else np.inf
        best_ranks.append({"best_gene_rank": best_gene_rank, "best_cluster_rank": best_cluster_rank})

    ranks_df = pd.DataFrame(best_ranks)
    n = len(ranks_df)
    if n == 0:
        return empty_result

    def _mrr(series):
        return series.apply(lambda x: 1.0 / x if np.isfinite(x) else 0.0).mean()

    def _topk(series, k):
        return float((series <= k).sum()) / n

    finite_gene = ranks_df["best_gene_rank"].replace(np.inf, np.nan)
    finite_cluster = ranks_df["best_cluster_rank"].replace(np.inf, np.nan)

    return {
        "method": method_name,
        "mrr_gene": _mrr(ranks_df["best_gene_rank"]),
        "mrr_cluster": _mrr(ranks_df["best_cluster_rank"]),
        "top1_gene": _topk(ranks_df["best_gene_rank"], 1),
        "top1_cluster": _topk(ranks_df["best_cluster_rank"], 1),
        "top5_gene": _topk(ranks_df["best_gene_rank"], 5),
        "top5_cluster": _topk(ranks_df["best_cluster_rank"], 5),
        "top10_gene": _topk(ranks_df["best_gene_rank"], 10),
        "top10_cluster": _topk(ranks_df["best_cluster_rank"], 10),
        "median_rank_gene": float(finite_gene.median()) if finite_gene.notna().any() else np.nan,
        "median_rank_cluster": float(finite_cluster.median()) if finite_cluster.notna().any() else np.nan,
        "all_tf_pert": int(all_tf_pert),
    }


METHOD_LABEL_MAP = {
    "GiMMeMotifs ROC (attribution_re2g_extended)": "Attribution + rE2G extended",
    "GiMMeMotifs ROC (attribution_re2g_extended_shuffle)": "Attribution + rE2G extended (shuffle)",

    "GiMMeMotifs ROC (attribution_re2g)": "Attribution + rE2G",
    "GiMMeMotifs ROC (attribution_re2g_shuffle)": "Attribution + rE2G (shuffle)",

    "GiMMeMotifs ROC (attribution_abc)": "Attribution + ABC score",
    "GiMMeMotifs ROC (attribution_abc_shuffle)": "Attribution + ABC score (shuffle)",

    "GiMMeMotifs ROC (attribution)": "Attribution",
    "GiMMeMotifs ROC (attribution_shuffle)": "Attribution (shuffle)",

    "GiMMeMotifs ROC (re2g_extended)": "rE2G extended",
    "GiMMeMotifs ROC (re2g_extended_shuffle)": "rE2G extended (shuffle)",

    "GiMMeMotifs ROC (re2g)": "rE2G",
    "GiMMeMotifs ROC (re2g_shuffle)": "rE2G (shuffle)",

    "GiMMeMotifs ROC (abc_score)": "ABC score",
    "GiMMeMotifs ROC (abc_score_shuffle)": "ABC score (shuffle)",

    "GiMMeMotifs ROC (fanta_bio)": "fanta.bio",
    "GiMMeMotifs ROC (fanta_bio_shuffle)": "fanta.bio (shuffle)",

    "GiMMeMotifs ROC (tss_1kbp)": "tss_1kbp",
}

METHOD_DISPLAY_ORDER = [
    "Attribution + rE2G extended", "Attribution + rE2G extended (shuffle)",
    "Attribution + rE2G", "Attribution + rE2G (shuffle)",
    "Attribution + ABC score", "Attribution + ABC score (shuffle)",
    "Attribution", "Attribution (shuffle)",
    "rE2G extended", "rE2G extended (shuffle)",
    "rE2G", "rE2G (shuffle)",
    "ABC score", "ABC score (shuffle)",
    "tss_1kbp",
]

METHOD_DISPLAY_ORDER_NOSHUFFLE = [
    "Attribution + rE2G extended",
    "Attribution + rE2G",
    "Attribution + ABC score",
    "Attribution",
    "rE2G extended",
    "rE2G",
    "ABC score",
    "tss_1kbp",
    "Attribution (shuffle)",
]

METHOD_DISPLAY_ORDER_NOINTEGRATION = [
    "Attribution", "Attribution (shuffle)",
    "rE2G extended", "rE2G extended (shuffle)",
    "rE2G", "rE2G (shuffle)",
    "ABC score", "ABC score (shuffle)",
    "tss_1kbp",
]

METHOD_DISPLAY_ORDER_NOINTEGRATION_NOSHUFFLE = [
    "Attribution",
    "rE2G extended",
    "rE2G",
    "ABC score",
    "tss_1kbp",
    "Attribution (shuffle)",
]


def _stats_to_plot_df(stats_list: List[Dict[str, object]]) -> pd.DataFrame:
    plot_data = []
    for stats in stats_list:
        total = float(stats["all_tf_pert"]) if stats["all_tf_pert"] else 0.0
        gene_ratio = float(stats["matched_gene"]) / total if total > 0 else 0.0
        cluster_ratio = float(stats["matched_cluster"]) / total if total > 0 else 0.0

        display_method = METHOD_LABEL_MAP.get(stats["method"], stats["method"])

        plot_data.append({
            "Method": display_method,
            "Threshold": stats["threshold"],
            "Match Type": "Gene match",
            "Ratio": gene_ratio,
            "Count": int(stats["matched_gene"]),
            "Total": int(stats["all_tf_pert"]),
        })
        plot_data.append({
            "Method": display_method,
            "Threshold": stats["threshold"],
            "Match Type": "Cluster match",
            "Ratio": cluster_ratio,
            "Count": int(stats["matched_cluster"]),
            "Total": int(stats["all_tf_pert"]),
        })
    return pd.DataFrame(plot_data)


def plot_grouped_comparison(
    stats_list: List[Dict[str, object]],
    study: str,
    output_filename: str = "matched_genes_ratio_grouped.svg",
    method_display_order: Optional[List[str]] = None,
    title: str = "GiMMeMotifs: Motif Analysis Comparison",
) -> pd.DataFrame:
    if method_display_order is None:
        method_display_order = METHOD_DISPLAY_ORDER

    plot_df = _stats_to_plot_df(stats_list)

    method_order = [m for m in method_display_order if m in plot_df["Method"].values]
    if not method_order:
        print(f"  [WARN] No methods to plot for {output_filename}")
        return plot_df

    FONTSIZE = 23
    fig, ax = plt.subplots(figsize=(36/2.54, 22/2.54), dpi=300)
    plt.rcParams["font.size"] = FONTSIZE
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=plot_df,
        x="Method",
        y="Ratio",
        hue="Match Type",
        palette="Dark2",
        hue_order=["Cluster match", "Gene match"],
        order=method_order,
        ax=ax
    )

    plt.xticks(rotation=45, ha="right", fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.ylim(0, 1.0)
    plt.ylabel("Ratio of matched\nperturbations", fontsize=FONTSIZE)
    plt.xlabel("", fontsize=FONTSIZE)
    plt.title(title, fontsize=FONTSIZE)
    plt.legend(loc="upper right", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"figures/{study}/gimmemotifs/{output_filename}")
    plt.close()

    return plot_df


def plot_rank_metrics(
    rank_stats_list: List[Dict[str, object]],
    study: str,
    output_filename: str = "rank_metrics.svg",
    method_display_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    if method_display_order is None:
        method_display_order = METHOD_DISPLAY_ORDER

    plot_data = []
    for rs in rank_stats_list:
        display_method = METHOD_LABEL_MAP.get(rs["method"], rs["method"])
        plot_data.append({"Method": display_method, "Metric": "MRR (Gene)", "Value": rs["mrr_gene"]})
        plot_data.append({"Method": display_method, "Metric": "MRR (Cluster)", "Value": rs["mrr_cluster"]})
        plot_data.append({"Method": display_method, "Metric": "Top-1 (Gene)", "Value": rs["top1_gene"]})
        plot_data.append({"Method": display_method, "Metric": "Top-1 (Cluster)", "Value": rs["top1_cluster"]})
        plot_data.append({"Method": display_method, "Metric": "Top-5 (Gene)", "Value": rs["top5_gene"]})
        plot_data.append({"Method": display_method, "Metric": "Top-5 (Cluster)", "Value": rs["top5_cluster"]})
        plot_data.append({"Method": display_method, "Metric": "Top-10 (Gene)", "Value": rs["top10_gene"]})
        plot_data.append({"Method": display_method, "Metric": "Top-10 (Cluster)", "Value": rs["top10_cluster"]})

    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        return plot_df

    method_order = [m for m in method_display_order if m in plot_df["Method"].values]
    if not method_order:
        print(f"  [WARN] No methods to plot for {output_filename}")
        return plot_df

    metric_pairs = [
        ("MRR (Gene)", "MRR (Cluster)", "Mean Reciprocal Rank"),
        ("Top-1 (Gene)", "Top-1 (Cluster)", "Top-1 Hit Rate"),
        ("Top-5 (Gene)", "Top-5 (Cluster)", "Top-5 Hit Rate"),
        ("Top-10 (Gene)", "Top-10 (Cluster)", "Top-10 Hit Rate"),
    ]
    FONTSIZE = 16.5
    fig, axes = plt.subplots(2, 2, figsize=(60/2.54, 40/2.54), dpi=300)
    plt.rcParams["font.size"] = FONTSIZE
    sns.set_theme(style="whitegrid")

    for ax, (gene_metric, cluster_metric, panel_title) in zip(axes.flat, metric_pairs):
        sub = plot_df[plot_df["Metric"].isin([gene_metric, cluster_metric])].copy()
        sub["Match Type"] = sub["Metric"].apply(lambda x: "Gene match" if "Gene" in x else "Cluster match")
        sns.barplot(
            data=sub,
            x="Method",
            y="Value",
            hue="Match Type",
            palette="Dark2",
            hue_order=["Cluster match", "Gene match"],
            order=method_order,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONTSIZE)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(panel_title, fontsize=FONTSIZE)
        ax.set_xlabel("", fontsize=FONTSIZE)
        ax.set_title(panel_title, fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE, loc="upper right")

    plt.suptitle("GiMMeMotifs: Rank-based Motif Recovery Metrics", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"figures/{study}/gimmemotifs/{output_filename}")
    plt.close()

    return plot_df


if __name__ == "__main__":

    cluster = load_cluster_info()

    print("\n" + "=" * 60)
    print("Computing rank-based metrics (MRR, top-k hit rate)...")
    print("=" * 60)

    gimme_roc_df_full, bed_type_pert_counts_full = load_gimme_roc_reports(
        study=study,
        tfs=tfs,
        bed_types=gimme_bed_types_roc,
        pval_threshold=None,
        top_n=None,
    )

    rank_stats_all: List[Dict[str, object]] = []
    if not gimme_roc_df_full.empty:
        gimme_roc_cluster_df_full = process_gimme_roc_with_cluster(gimme_roc_df_full, cluster)

        for bed_type in gimme_bed_types_roc:
            sub = gimme_roc_cluster_df_full[gimme_roc_cluster_df_full["bed_type"] == bed_type]
            n_perts = bed_type_pert_counts_full.get(bed_type, 0)
            tfs_with_file = sorted(sub["perturbation"].unique()) if not sub.empty else []

            rstats = calculate_rank_stats(
                cluster_df=sub,
                method_name=f"GiMMeMotifs ROC ({bed_type})",
                all_tf_pert=n_perts,
                tfs_with_file=tfs_with_file,
            )
            rank_stats_all.append(rstats)
            print(f"  {bed_type}: MRR(gene)={rstats['mrr_gene']:.3f}  MRR(cluster)={rstats['mrr_cluster']:.3f}  "
                  f"Top-1(gene)={rstats['top1_gene']:.3f}  Top-5(gene)={rstats['top5_gene']:.3f}  "
                  f"median_rank(gene)={rstats['median_rank_gene']}")

    rank_stats_df = pd.DataFrame(rank_stats_all)
    if not rank_stats_df.empty:
        rank_stats_df["method_label"] = rank_stats_df["method"].map(METHOD_LABEL_MAP)
        rank_stats_df.to_csv(f"figures/{study}/gimmemotifs/rank_metrics_summary.txt", sep="\t", index=False)
        print(f"\nSaved: figures/{study}/gimmemotifs/rank_metrics_summary.txt")

        plot_rank_metrics(rank_stats_all, study, "rank_metrics_all.svg", METHOD_DISPLAY_ORDER)
        print(f"  Saved to: figures/{study}/gimmemotifs/rank_metrics_all.svg")

        plot_rank_metrics(rank_stats_all, study, "rank_metrics_noshuffle.svg", METHOD_DISPLAY_ORDER_NOSHUFFLE)
        print(f"  Saved to: figures/{study}/gimmemotifs/rank_metrics_noshuffle.svg")

        plot_rank_metrics(rank_stats_all, study, "rank_metrics_nointegration.svg", METHOD_DISPLAY_ORDER_NOINTEGRATION)
        print(f"  Saved to: figures/{study}/gimmemotifs/rank_metrics_nointegration.svg")

        plot_rank_metrics(rank_stats_all, study, "rank_metrics_nointegration_noshuffle.svg", METHOD_DISPLAY_ORDER_NOINTEGRATION_NOSHUFFLE)
        print(f"  Saved to: figures/{study}/gimmemotifs/rank_metrics_nointegration_noshuffle.svg")

    threshold_specs: List[Tuple[str, float]] = [
        ("pval", gimme_roc_pval_threshold),
        ("pval", 0.01),
        ("pval", 0.05),
        ("qval", 0.05),  # BH-FDR corrected p-value <= 0.05
    ]

    for thr_mode, thr_val in threshold_specs:
        if thr_mode == "qval":
            run_label = f"q{thr_val:.0e}"
            threshold_str = f"q<={thr_val}"
            pval_thr_arg: Optional[float] = None
            qval_thr_arg: Optional[float] = thr_val
        else:
            run_label = f"p{thr_val:.0e}" if thr_val is not None else "pAll"
            threshold_str = f"p<={thr_val}"
            pval_thr_arg = thr_val
            qval_thr_arg = None

        print(f"\n{'='*60}")
        print(f"Processing with {thr_mode}-threshold: {thr_val}  (run_label={run_label})")
        print(f"{'='*60}")

        stats_all: List[Dict[str, object]] = []

        print("\nLoading GiMMeMotifs results (gimme.roc.report.txt)...")
        gimme_roc_df, bed_type_pert_counts = load_gimme_roc_reports(
            study=study,
            tfs=tfs,
            bed_types=gimme_bed_types_roc,
            pval_threshold=pval_thr_arg,
            top_n=None,
            qval_threshold=qval_thr_arg,
        )

        processed_bed_types: set = set()

        if not gimme_roc_df.empty:
            if DEBUG:
                print("\n[DEBUG] ===== Source file summary =====")
                for bed_type in sorted(gimme_roc_df["bed_type"].unique()):
                    sub = gimme_roc_df[gimme_roc_df["bed_type"] == bed_type]
                    unique_files = sub["source_file"].unique()
                    print(f"[DEBUG] bed_type={bed_type}: {len(sub)} records from {len(unique_files)} files")
                    for f in unique_files[:3]:
                        print(f"  [DEBUG]   - {f}")
                print("[DEBUG] ===================================\n")

            gimme_roc_cluster_df = process_gimme_roc_with_cluster(gimme_roc_df, cluster)

            for bed_type in sorted(gimme_roc_cluster_df["bed_type"].unique()):
                sub = gimme_roc_cluster_df[gimme_roc_cluster_df["bed_type"] == bed_type].copy()

                n_perts_with_file = bed_type_pert_counts.get(bed_type, len(set(sub["perturbation"])))

                stats_roc = calculate_stats(
                    cluster_df=sub,
                    method_name=f"GiMMeMotifs ROC ({bed_type})",
                    threshold_type=threshold_str,
                    all_tf_pert=n_perts_with_file
                )
                stats_all.append(stats_roc)
                processed_bed_types.add(bed_type)

                print(f"  ROC({bed_type}): {stats_roc['matched_gene']} gene matches, {stats_roc['matched_cluster']} cluster matches out of {stats_roc['all_tf_pert']}")

                if DEBUG:
                    gene_matched_perts = sorted(set(sub.query("gene_match_flag == 1")["perturbation"]))
                    cluster_matched_perts = sorted(set(sub.query("cluster_match_flag == 1")["perturbation"]))
                    if gene_matched_perts:
                        print(f"    [DEBUG] Gene matches: {', '.join(gene_matched_perts)}")
                    if cluster_matched_perts:
                        print(f"    [DEBUG] Cluster matches: {', '.join(cluster_matched_perts)}")

            gimme_roc_df.to_csv(f"figures/{study}/gimmemotifs/gimme_roc_extracted_motifs_{run_label}.txt", sep="\t", index=False)
            gimme_roc_cluster_df.to_csv(f"figures/{study}/gimmemotifs/gimme_roc_motif_match_{run_label}.txt", sep="\t", index=False)
            print(f"\n  Saved: figures/{study}/gimmemotifs/gimme_roc_extracted_motifs_{run_label}.txt")
            print(f"  Saved: figures/{study}/gimmemotifs/gimme_roc_motif_match_{run_label}.txt")

        else:
            print("  No gimme.roc.report.txt parsed results found")

        for bed_type in gimme_bed_types_roc:
            if bed_type in bed_type_pert_counts and bed_type not in processed_bed_types:
                n_perts = bed_type_pert_counts[bed_type]
                stats_all.append({
                    "method": f"GiMMeMotifs ROC ({bed_type})",
                    "threshold": threshold_str,
                    "matched_gene": 0,
                    "matched_cluster": 0,
                    "all_tf_pert": int(n_perts),
                })
                print(f"  ROC({bed_type}): 0 matches (all filtered by {threshold_str}), {n_perts} perts had files -> plotted as 0%")

        stats_summary = pd.DataFrame(stats_all)
        if not stats_summary.empty:
            stats_summary["gene_ratio"] = stats_summary["matched_gene"] / stats_summary["all_tf_pert"].replace(0, np.nan)
            stats_summary["cluster_ratio"] = stats_summary["matched_cluster"] / stats_summary["all_tf_pert"].replace(0, np.nan)
            stats_summary = stats_summary.fillna(0.0)
            stats_summary.to_csv(f"figures/{study}/gimmemotifs/motif_analysis_summary_{run_label}.txt", sep="\t", index=False)
            print(f"\nSaved summary: figures/{study}/gimmemotifs/motif_analysis_summary_{run_label}.txt")

            print(f"\nGenerating plot ({thr_mode}={thr_val})...")
            title_suffix = " [BH-FDR]" if thr_mode == "qval" else ""
            plot_grouped_comparison(
                stats_all, study,
                output_filename=f"matched_genes_ratio_{run_label}.svg",
                title=f"GiMMeMotifs: Motif Analysis Comparison (with shuffle controls){title_suffix}",
            )
            print(f"  Saved to: figures/{study}/gimmemotifs/matched_genes_ratio_{run_label}.svg")

            plot_grouped_comparison(
                stats_all, study,
                output_filename=f"matched_genes_ratio_{run_label}_noshuffle.svg",
                method_display_order=METHOD_DISPLAY_ORDER_NOSHUFFLE,
                title=f"GiMMeMotifs: Motif Analysis Comparison{title_suffix}",
            )
            print(f"  Saved to: figures/{study}/gimmemotifs/matched_genes_ratio_{run_label}_noshuffle.svg")

            plot_grouped_comparison(
                stats_all, study,
                output_filename=f"matched_genes_ratio_{run_label}_nointegration.svg",
                method_display_order=METHOD_DISPLAY_ORDER_NOINTEGRATION,
                title=f"GiMMeMotifs: Motif Analysis Comparison (no integration, with shuffle){title_suffix}",
            )
            print(f"  Saved to: figures/{study}/gimmemotifs/matched_genes_ratio_{run_label}_nointegration.svg")

            plot_grouped_comparison(
                stats_all, study,
                output_filename=f"matched_genes_ratio_{run_label}_nointegration_noshuffle.svg",
                method_display_order=METHOD_DISPLAY_ORDER_NOINTEGRATION_NOSHUFFLE,
                title=f"GiMMeMotifs: Motif Analysis Comparison (no integration){title_suffix}",
            )
            print(f"  Saved to: figures/{study}/gimmemotifs/matched_genes_ratio_{run_label}_nointegration_noshuffle.svg")

    print("\nDone!")
