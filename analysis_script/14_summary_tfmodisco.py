# conda activate enformer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


study_name   = "NormanWeissman2019_filtered_mixscape_exnp_train"
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study        = f'{study_name}__{study_suffix}'
pretraind_model = "enformer"
modisco_suffix = "_short_fdr02"
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


df[['motif_id', 'motif_gene']] = df['match'].str.split('_', expand=True)
df['pert_gene'] = df['perturbation'].str.split('.').str[1]

## split genes ##
split_genes = df.set_index(["match"])["motif_gene"].str.split('::', expand=True)
split_genes = split_genes.stack().reset_index(level=1, drop=True).str.upper()
split_genes.name = 'motif_gene'
df = pd.merge(df.drop('motif_gene', axis=1), split_genes.reset_index().drop_duplicates(), on="match")
split_genes2 = df.set_index("perturbation")["pert_gene"].str.split('_', expand=True)
split_genes2 = split_genes2.stack().reset_index(level=1, drop=True).str.upper()
split_genes2.name = 'pert_gene'
df = pd.merge(df.drop('pert_gene', axis=1), split_genes2.reset_index().drop_duplicates(), on="perturbation")
df = df.drop_duplicates()

### cluster ###
cluster = pd.read_csv("reference/jaspar/clusters.tab", sep="\t", usecols=[0,2])
cluster = cluster.set_index("cluster")["name"].str.split(",",expand=True).stack().str.upper().reset_index().rename(columns={0:"cluster_gene"}).drop("level_1", axis=1)
cluster_df = pd.merge(df,         cluster.rename(columns={"cluster":"motif_cluster"}), left_on="motif_gene", right_on="cluster_gene", how="left")
cluster_df = pd.merge(cluster_df, cluster.rename(columns={"cluster":"pert_cluster"}),  left_on="pert_gene",  right_on="cluster_gene", how="left")
cluster_df = cluster_df.drop_duplicates()


## check match ##
cluster_df['gene_match_flag'] = (cluster_df['motif_gene'] == cluster_df['pert_gene']).astype(int)
cluster_df['cluster_match_flag'] = (cluster_df['motif_cluster'] == cluster_df['pert_cluster']).astype(int)

## stats ##
cluster_dfth = cluster_df.query('rank < 10')
num_match_gene  = len(set(cluster_dfth.query('gene_match_flag == 1')["perturbation"]))
num_match_clst  = len(set(cluster_dfth.query('cluster_match_flag == 1')["perturbation"]))
num_tf_pert     = len(tfs)
num_tf_pert_wm  = len(set(split_genes2[split_genes2.isin(list(set(split_genes) & set(cluster["cluster_gene"])))].index))

result = pd.DataFrame([[num_match_gene, num_match_clst, num_tf_pert_wm]], columns=["matched_gene", "matched_cluster", "all_tf_pert"]).T
result.to_csv(f"figures/{study}/tfmodisco/tfmodisco_motif_match_num.txt", sep="\t", header=False, index=False)
cluster_df.to_csv(f"figures/{study}/tfmodisco/tfmodisco_motif_match.txt", sep="\t", index=False)

## stats qval ##
cluster_df04 = cluster_df.query('qval < 0.4')
num_match_gene  = len(set(cluster_df04.query('gene_match_flag == 1')["perturbation"]))
num_match_clst  = len(set(cluster_df04.query('cluster_match_flag == 1')["perturbation"]))
num_tf_pert     = len(tfs)
num_tf_pert_wm  = len(set(split_genes2[split_genes2.isin(list(set(split_genes) & set(cluster["cluster_gene"])))].index))

result04 = pd.DataFrame([[num_match_gene, num_match_clst, num_tf_pert_wm]], columns=["matched_gene", "matched_cluster", "all_tf_pert"]).T
result04.to_csv(f"figures/{study}/tfmodisco/tfmodisco_motif_match_num_04.txt", sep="\t", header=False, index=False)


## barplot ##
result = result / result.loc["all_tf_pert", 0]
result04 = result04 / result04.loc["all_tf_pert", 0]
result = result.rename(columns={0:"Ratio"})
result04 = result04.rename(columns={0:"Ratio"})

result["Threshold"] = "top 10"
result04["Threshold"] = "q-value < 0.4"

plot_df = pd.concat([result.reset_index(), result04.reset_index()]).query('index != "all_tf_pert"')
plot_df["index"] = plot_df["index"].replace({"matched_gene":"Gene match", "matched_cluster":"Cluster match"})

def plot_barplot(plot_df, study):
    plt.figure(figsize=(14/2.54, 10/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    sns.set_theme(style="whitegrid")
    g = sns.barplot(
        data=plot_df,
        x="Threshold",
        y="Ratio",
        hue="index",
        palette="Dark2",
        order=["top 10", "q-value < 0.4"],
        hue_order=["Cluster match", "Gene match"],
    )
    plt.xticks(rotation=45)
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(f"figures/{study}/tfmodisco/matched_genes_ratio.svg")


plot_barplot(plot_df, study)

