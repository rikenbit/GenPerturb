import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) >= 3:
    study_name   = sys.argv[1]
    study_suffix = sys.argv[2]
else:
    study_name   = "NormanWeissman2019_filtered_mixscape_exnp_train"
    study_suffix = "borzoi_transfer_epoch100_batch256_adamw5e3"

study        = f'{study_name}__{study_suffix}'
pretraind_model = study_suffix.split("_")[0]
modisco_suffix = ""
suffix=""
os.makedirs(f'figures/{study}/tfmodisco', exist_ok=True)

tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].to_list()
tfs = [i for i in os.listdir(f"tfmodisco/{study}/") if any(j in i for j in tf_list)]


def add_rank(df):
    pattern_dfs = pd.DataFrame()
    for pattern in df['pattern'].unique():
        pattern_df = df[df['pattern'] == pattern]
        pattern_df = pattern_df.sort_values(by='qval')
        pattern_df['rank'] = pattern_df['qval'].rank(method='dense')
        pattern_dfs = pd.concat([pattern_dfs, pattern_df])
    return pattern_dfs


df = pd.DataFrame()

for pert in tfs:
    try:
        report_dir = f"tfmodisco/{study}/{pert}{suffix}/modisco_result{modisco_suffix}"
        each_df = pd.read_csv(f"{report_dir}/{pert}_MA_list.txt", sep="\t")
        each_df = add_rank(each_df)
        df = pd.concat([df, each_df])
    except:
        continue

df = df.sort_values(["perturbation", "qval"]).reset_index(drop=True)
df["global_rank"] = df.groupby("perturbation").cumcount() + 1

df['motif_id'] = df['match'].str.split('_').str[0]
df['motif_gene'] = df['match'].str.split('_').str[1].str.split('.', n=2).str[2]
df['pert_gene'] = df['perturbation'].str.split('.').str[1]

split_genes = df.set_index(["match"])["motif_gene"].str.split('::', expand=True)
split_genes = split_genes.stack().reset_index(level=1, drop=True).str.upper()
split_genes.name = 'motif_gene'
df = pd.merge(df.drop('motif_gene', axis=1), split_genes.reset_index().drop_duplicates(), on="match")
split_genes2 = df.set_index("perturbation")["pert_gene"].str.split('_', expand=True)
split_genes2 = split_genes2.stack().reset_index(level=1, drop=True).str.upper()
split_genes2.name = 'pert_gene'
df = pd.merge(df.drop('pert_gene', axis=1), split_genes2.reset_index().drop_duplicates(), on="perturbation")
df = df.drop_duplicates()

cluster = pd.read_csv("reference/jaspar/clusters.tab", sep="\t", usecols=[0,2])
cluster = cluster.set_index("cluster")["name"].str.split(",",expand=True).stack().str.upper().reset_index().rename(columns={0:"cluster_gene"}).drop("level_1", axis=1)
cluster = cluster.set_index("cluster")["cluster_gene"].str.split("::",expand=True).stack().str.strip().reset_index().rename(columns={0:"cluster_gene"}).drop("level_1", axis=1)
cluster = cluster.drop_duplicates()
cluster_df = pd.merge(df,         cluster.rename(columns={"cluster":"motif_cluster"}), left_on="motif_gene", right_on="cluster_gene", how="left")
cluster_df = pd.merge(cluster_df, cluster.rename(columns={"cluster":"pert_cluster"}),  left_on="pert_gene",  right_on="cluster_gene", how="left")
cluster_df = cluster_df.drop_duplicates()


cluster_df['gene_match_flag'] = (cluster_df['motif_gene'] == cluster_df['pert_gene']).astype(int)
cluster_df['cluster_match_flag'] = (cluster_df['motif_cluster'] == cluster_df['pert_cluster']).astype(int)

gene_cols    = ["perturbation", "pert_gene", "match", "motif_gene", "pattern", "num_seqlets", "qval", "rank", "global_rank"]
cluster_cols = ["perturbation", "pert_gene", "match", "motif_gene", "pattern", "num_seqlets", "qval", "rank", "global_rank", "motif_cluster", "pert_cluster"]

cluster_df.query('gene_match_flag == 1')[gene_cols].drop_duplicates().sort_values(["perturbation", "qval"]).to_csv(
    f"figures/{study}/tfmodisco/tfmodisco_gene_match.txt", sep="\t", index=False)
cluster_df.query('cluster_match_flag == 1')[cluster_cols].drop_duplicates().sort_values(["perturbation", "qval"]).to_csv(
    f"figures/{study}/tfmodisco/tfmodisco_cluster_match.txt", sep="\t", index=False)


def calculate_rank_stats(cluster_df, all_perts):
    best_ranks = []
    for pert in all_perts:
        sub = cluster_df[cluster_df["perturbation"] == pert]
        if sub.empty:
            best_ranks.append({"best_gene_rank": np.inf, "best_cluster_rank": np.inf})
            continue
        gene_matches = sub[sub["gene_match_flag"] == 1]
        cluster_matches = sub[sub["cluster_match_flag"] == 1]
        best_gene_rank = gene_matches["global_rank"].min() if not gene_matches.empty else np.inf
        best_cluster_rank = cluster_matches["global_rank"].min() if not cluster_matches.empty else np.inf
        best_ranks.append({"best_gene_rank": best_gene_rank, "best_cluster_rank": best_cluster_rank})

    ranks_df = pd.DataFrame(best_ranks)
    n = len(ranks_df)
    if n == 0:
        return {}

    def _mrr(s):
        return s.apply(lambda x: 1.0 / x if np.isfinite(x) else 0.0).mean()

    def _topk(s, k):
        return float((s <= k).sum()) / n

    finite_gene = ranks_df["best_gene_rank"].replace(np.inf, np.nan)
    finite_cluster = ranks_df["best_cluster_rank"].replace(np.inf, np.nan)

    return {
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
        "all_tf_pert": n,
    }


all_perts = sorted(cluster_df["perturbation"].unique())
rank_stats = calculate_rank_stats(cluster_df, all_perts)

rank_stats_df = pd.DataFrame([rank_stats])
rank_stats_df.to_csv(f"figures/{study}/tfmodisco/rank_metrics_summary.txt", sep="\t", index=False)
print(f"Rank metrics: MRR(gene)={rank_stats['mrr_gene']:.3f}  MRR(cluster)={rank_stats['mrr_cluster']:.3f}  "
      f"Top-1(gene)={rank_stats['top1_gene']:.3f}  Top-5(gene)={rank_stats['top5_gene']:.3f}  "
      f"Top-10(gene)={rank_stats['top10_gene']:.3f}  median_rank(gene)={rank_stats['median_rank_gene']}")
print(f"Saved: figures/{study}/tfmodisco/rank_metrics_summary.txt")


qval_thresholds = [0.05, 0.01, 0.001]
rank_stats_by_qval = []
for qval_thr in qval_thresholds:
    filtered = cluster_df[cluster_df["qval"] < qval_thr]
    rs = calculate_rank_stats(filtered, all_perts)
    rs["threshold"] = f"q<{qval_thr}"
    rank_stats_by_qval.append(rs)
    print(f"  q<{qval_thr}: MRR(gene)={rs['mrr_gene']:.3f}  Top-5(gene)={rs['top5_gene']:.3f}")

rs_all = calculate_rank_stats(cluster_df, all_perts)
rs_all["threshold"] = "All motifs"
rank_stats_by_qval.insert(0, rs_all)


num_tf_pert_wm  = df['perturbation'].nunique()

def calc_result(cluster_df_filtered):
    n_gene = len(set(cluster_df_filtered.query('gene_match_flag == 1')["perturbation"]))
    n_clst = len(set(cluster_df_filtered.query('cluster_match_flag == 1')["perturbation"]))
    return pd.DataFrame([[n_gene, n_clst, num_tf_pert_wm]], columns=["matched_gene", "matched_cluster", "all_tf_pert"]).T

result005 = calc_result(cluster_df.query('qval < 0.05'))
result001 = calc_result(cluster_df.query('qval < 0.01'))


def normalize(r, label):
    r = r / r.loc["all_tf_pert", 0]
    r = r.rename(columns={0: "Ratio"})
    r["Threshold"] = label
    return r

result005 = normalize(result005, "q-value < 0.05")
result001 = normalize(result001, "q-value < 0.01")

plot_df = pd.concat([result005.reset_index(), result001.reset_index()]).query('index != "all_tf_pert"')
plot_df["index"] = plot_df["index"].replace({"matched_gene": "Gene match", "matched_cluster": "Cluster match"})

def plot_barplot(plot_df, study, output_filename="matched_genes_ratio.svg",
                 title="TF-MoDISco Motif Match Ratio", threshold_order=None,
                 figsize=None):
    if figsize is None:
        figsize = (14/2.54, 10/2.54)
    plt.figure(figsize=figsize, dpi=300)
    plt.rcParams["font.size"] = 6
    sns.set_theme(style="whitegrid")
    if threshold_order is None:
        threshold_order = ["q-value < 0.05", "q-value < 0.01"]
    g = sns.barplot(
        data=plot_df,
        x="Threshold",
        y="Ratio",
        hue="index",
        palette="Dark2",
        order=threshold_order,
        hue_order=["Cluster match", "Gene match"],
    )
    plt.xticks(rotation=45)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(0, 1.05)
    plt.xlabel("")
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(f"figures/{study}/tfmodisco/{output_filename}")
    plt.close()


plot_barplot(plot_df, study)

plot_df_005 = plot_df[plot_df["Threshold"] == "q-value < 0.05"]
plot_df_005 = plot_df_005.copy()
plot_df_005["Threshold"] = "Attribution"
plot_barplot(plot_df_005, study, output_filename="matched_genes_ratio_q005.svg",
             title="TF-MoDISco Motif Match Ratio\n(q < 0.05)",
             threshold_order=["Attribution"],
             figsize=(14*2/3/2.54, 10/2.54))


def plot_rank_metrics(rank_stats_by_qval, study, output_filename="rank_metrics.svg",
                      title="TF-MoDISco Rank-based Motif Recovery",
                      threshold_filter=None):
    plot_data = []
    for rs in rank_stats_by_qval:
        thr = rs["threshold"]
        if threshold_filter is not None and thr not in threshold_filter:
            continue
        plot_data.append({"Threshold": thr, "Metric": "MRR", "Match Type": "Gene match", "Value": rs["mrr_gene"]})
        plot_data.append({"Threshold": thr, "Metric": "MRR", "Match Type": "Cluster match", "Value": rs["mrr_cluster"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-1", "Match Type": "Gene match", "Value": rs["top1_gene"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-1", "Match Type": "Cluster match", "Value": rs["top1_cluster"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-5", "Match Type": "Gene match", "Value": rs["top5_gene"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-5", "Match Type": "Cluster match", "Value": rs["top5_cluster"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-10", "Match Type": "Gene match", "Value": rs["top10_gene"]})
        plot_data.append({"Threshold": thr, "Metric": "Top-10", "Match Type": "Cluster match", "Value": rs["top10_cluster"]})

    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        return

    threshold_order = ["All motifs"] + [f"q<{q}" for q in [0.05, 0.01, 0.001]]
    threshold_order = [t for t in threshold_order if t in plot_df["Threshold"].values]

    metric_list = ["MRR", "Top-1", "Top-5", "Top-10"]
    fig, axes = plt.subplots(1, 4, figsize=(36/2.54, 10/2.54), dpi=300)
    plt.rcParams["font.size"] = 7
    sns.set_theme(style="whitegrid")

    for ax, metric in zip(axes, metric_list):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(
            data=sub,
            x="Threshold",
            y="Value",
            hue="Match Type",
            palette="Dark2",
            hue_order=["Cluster match", "Gene match"],
            order=threshold_order,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.set_title(f"TF-MoDISco {metric}")
        ax.legend(fontsize=5, loc="upper right")

    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"figures/{study}/tfmodisco/{output_filename}")
    plt.close()
    print(f"Saved: figures/{study}/tfmodisco/{output_filename}")


plot_rank_metrics(rank_stats_by_qval, study)

plot_rank_metrics(rank_stats_by_qval, study,
                  output_filename="rank_metrics_q005.svg",
                  title="TF-MoDISco Rank-based Motif Recovery (q < 0.05)",
                  threshold_filter={"All motifs", "q<0.05"})
print("\nDone!")
