import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys



#study_name = sys.argv[1]
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
hdf5  = f'data/{study_name}.h5'
bed   = f'fasta/{study_name}.bed'
fasta = f'fasta/GRCh38.p13.genome.fa'
context_length=196_608

#study_suffix = sys.argv[2]
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'

pred = np.load(f"prediction/{study}/prediction.npy")
df2 = pd.DataFrame(pred)
df2.columns = df.columns[1:]
df2.index = df.index

ctrl = df2.columns[0]
df3 = (df2.T - df2[ctrl]).T.drop(ctrl, axis=1).copy()

df2 = df2.iloc[:,1:]


attribution = "ixg"
gene = "RINL"


def read_bed_files(study, gene, attribution, value="cpm"):
    bed_path = f"attribution_seq/{study}/{gene}/00_all_{gene}_ixg_{value}.bed"
    peak = pd.read_csv(bed_path, sep="\t", names=["chr", "start", "end", "gene", "strand", "score", "pert", "attr", f"peak"])
    merged_df = peak.pivot(index=["chr", "start", "end"], columns="pert", values="attr")
    merged_fl = peak.query('(peak == 1) | (peak == -1)').groupby(["chr", "start", "end"])["peak"].count()
    return merged_df, merged_fl

merged_df, merged_fl = read_bed_files(study, gene, attribution, value="cpm")
merged_df_fc, merged_fl_fc = read_bed_files(study, gene, attribution, value="fc")


def generate_correlation_histogram(merged_df, merged_fl, _df, gene, study, value="cpm", pert_num="all"):
    if pert_num == "all":
        filtered_df = merged_df.loc[merged_fl.index]
    elif pert_num == "top":
        top_perts = _df.loc[gene, :].abs().sort_values(ascending=False).head(50).index.to_list()
        merged_fl_top = merged_fl.T[merged_fl.columns.str.contains("|".join(top_perts))].T
        filtered_df = merged_df[(merged_fl_top != 0).any(axis=1)]
    correlation_df = filtered_df.apply(lambda row: row.corr(_df.loc[gene, filtered_df.columns]), axis=1)
    os.makedirs(f'figures/{study}/attribution/histogram', exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.rcParams["font.size"] = 12
    sns.histplot(correlation_df, bins=40, kde=True, color='skyblue')
    plt.title(gene)
    if value == "cpm":
        plt.xlabel('Correlation Coefficient (expression vs. attribution on peaks)')
    elif value == "fc":
        plt.xlabel('Correlation Coefficient (foldchange vs. attribution on peaks)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if pert_num == "all":
        plt.savefig(f'figures/{study}/attribution/histogram/cor_attr_{value}_{gene}.png')
    elif pert_num == "top":
        plt.savefig(f'figures/{study}/attribution/histogram/cor_attr_{value}_{gene}_top50.png')
    plt.clf()
    plt.close()
    return correlation_df

correlation_df = generate_correlation_histogram(merged_df, merged_fl, df2, gene, study, value="cpm", pert_num="all")
correlation_df_fc = generate_correlation_histogram(merged_df_fc, merged_fl_fc, df3, gene, study, value="fc",  pert_num="all")


## scatter plot ##
def create_scatter_plot(correlation_df, _df, gene, study, value="cpm"):
    top_genome_pos = correlation_df.sort_values().index[[0, len(correlation_df)-1]]
    for i in top_genome_pos:
        rep_attr = merged_df.loc[i]
        genome_pos = "_".join(str(i) for i in rep_attr.name)
        plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
        plt.rcParams["font.size"] = 5
        plt.rcParams["axes.titlesize"] = 5
        plt.scatter(x=_df.loc[gene, :], y=rep_attr, s=3, alpha=0.8)
        plt.title(f'{gene} ({genome_pos})')
        if value == "cpm":
            plt.xlabel("Expression")
        elif value == "fc":
            plt.xlabel("Fold change")
        plt.ylabel("Attribution")
        plt.tight_layout()
        plt.savefig(f'figures/{study}/attribution/scatter/{value}_{gene}_{genome_pos}.svg')
        plt.clf()
        plt.close()

os.makedirs(f'figures/{study}/attribution/scatter', exist_ok=True)
create_scatter_plot(correlation_df, df2, gene, study, value="cpm")
create_scatter_plot(correlation_df_fc, df3, gene, study, value="fc")


## heatmap ##
def create_cluster_heatmap(merged_df, _df, gene, study, value="cpm"):
    os.makedirs(f'figures/{study}/attribution/heatmap', exist_ok=True)
    merged_df.index.names = [merged_df.index[0][0], str(merged_df.index[0][1]), str(merged_df.index[-1][2])]
    merged_df = merged_df[_df.loc[gene, :].sort_values(ascending=False).index.to_list()]
    genome_pos = "_".join(str(i) for i in merged_df.index[0][:2]) + "_" + str(merged_df.index[-1][2])
    plt.figure(figsize=(20, 5), dpi=400)
    plt.rcParams["font.size"] = 18
    vmin = merged_df.quantile(0.05).min()
    vmax = merged_df.quantile(0.95).max()
    sns.clustermap(merged_df.T, cmap="RdBu_r", center=0, col_cluster=False, row_cluster=False,
                   vmin=vmin, vmax=vmax, xticklabels=False)
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.savefig(f'figures/{study}/attribution/heatmap/attr_{gene}_{value}.png')
    plt.clf()
    plt.close()

create_cluster_heatmap(merged_df, df2, gene, study, value="cpm")
create_cluster_heatmap(merged_df_fc, df3, gene, study, value="fc")


