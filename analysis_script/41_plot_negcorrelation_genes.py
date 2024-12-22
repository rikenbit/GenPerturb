# conda activate enformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


def plot_scatter_real_pred(df, df2, genes):
    real = df.loc[genes, :].T
    pred = df2.loc[genes, :].T
    fig, axes = plt.subplots(1, len(genes), figsize=(4.6/2.54 * len(genes), 4.5/2.54), dpi=300)
    for i in range(len(genes)):
        sns.scatterplot(x=real[genes[i]], y=pred[genes[i]], ax=axes[i], c="slateblue", s=20)
        correlation = np.corrcoef(real[genes[i]], pred[genes[i]])[0, 1]
        axes[i].set_xlabel(f"Real expression ({genes[i]})")
        axes[i].set_ylabel(f"Pred expression ({genes[i]})")
        axes[i].annotate(str(f"r = {correlation:.3f}"), xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=6, ha='left', va='top', color='black')
    plt.rcParams["font.size"] = 6
    plt.tight_layout()
    plt.savefig(f'figures/{study}/neg_cor/scatter_{"_".join(genes)}.svg')
    plt.clf()
    plt.close()


def calculate_pairwise_correlations(df1, df2):
    corr_df = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col1 in df1.columns:
        for col2 in df2.columns:
            corr = df1[col1].corr(df2[col2])
            corr_df.loc[col1, col2] = corr
    return corr_df


def correlation_heatmap(df, df2, genes):
    real = df.loc[genes, :].T.corr().round(2)
    pred = df2.loc[genes, :].T.corr().round(2)
    mask = np.triu(np.ones_like(real, dtype=bool), k=1)
    corr_df = calculate_pairwise_correlations(df.loc[genes, :].T, df2.loc[genes, :].T).astype("float").round(2)
    fig, axes = plt.subplots(1, 3, figsize=(3.7/2.54 * len(genes), 1.1/2.54 * len(genes)), dpi=300)
    plt.rcParams["font.size"] = 6.5
    for i,dat,title in zip([0,1], [real, pred], ["real", "pred"]):
        sns.heatmap(dat, ax=axes[i], annot=True, linewidths=.5, center=0,
            vmax=1, vmin=-1, cmap="magma")
            #vmax=1, vmin=-1, cmap="magma", mask=mask)
        axes[i].set_title(f"Correlation ({title})")
        for label in axes[i].get_xticklabels():
            label.set_rotation(90)
        for label in axes[i].get_yticklabels():
            label.set_rotation(0)
    sns.heatmap(corr_df, ax=axes[2], annot=True, linewidths=.5, center=0,
        vmax=1, vmin=-1, cmap="magma")
    for label in axes[2].get_xticklabels():
        label.set_rotation(90)
    for label in axes[2].get_yticklabels():
        label.set_rotation(0)
    axes[2].set_title(f"Correlation (real - pred)")
    plt.tight_layout()
    plt.savefig(f'figures/{study}/neg_cor/heatmap_{"_".join(genes)}.svg')
    plt.clf()
    plt.close()



study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
pretrained_models = ["enformer", "hyena_dna_tss", "hyena_dna_last", "nucleotide_transformer_tss", "nucleotide_transformer_cls"]
model_data = {}
for pretrained_model in pretrained_models:
    study_suffixes = [
        f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3_masked",
        f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3",
    ]
    df_paths = [
        f'data/{study_name}_enformer_masked.tsv',
        f'data/{study_name}.tsv',
    ]
    for df_path, study_suffix in zip(df_paths, study_suffixes):
        study = f'{study_name}__{study_suffix}'
        model_data[pretrained_model] = {
            "df_path": df_path,
            "study_suffix": study_suffix,
        }





## find focused genes (manual curation) ##
study_suffix = f"enformer_transfer_epoch100_batch256_adamw5e3"
bed   = pd.read_csv(f'fasta/{study_name}.bed', sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
cor = pd.read_csv(f'figures/{study_name}__{study_suffix}/cor_matrix/correlation_across_perts.txt' , sep="\t")
merged = pd.merge(cor, bed, on=["Gene", "training"], how="left")
merged_sort = merged.sort_values(["chr", "start", "end"]).reset_index(drop=False)
for i in merged.query('training == "test"').tail(5)["Gene"]:
    index = merged_sort[merged_sort["Gene"] == i].index[0]
    merged_sort[index-12:index+12]
    merged_sort[index-12:index+12]["Gene"].to_list()    







## plot ##
genes_list = [
['PSMD8', 'SPRED3', 'FAM98C', 'RASGRP4', 'RYR1', 'MAP4K1', 'EIF3K', 'ACTN4', 'CAPN12', 'ECH1', 'HNRNPL', 'RINL', 'NFKBIB', 'SIRT2', 'CCER2', 'MRPS12', 'SARS2', 'FBXO17', 'FBXO27', 'PAK4', 'LRFN1', 'GMFG', 'SAMD4B', 'PAF1'],
#['ECH1', 'HNRNPL', 'RINL', 'NFKBIB', 'SIRT2', 'CCER2', 'MRPS12', 'SARS2', 'FBXO17'],
#['ZNF579', 'ZNF524', 'FIZ1', 'ZNF865', 'ZNF784', 'ZNF580', 'ZNF581', 'CCDC106', 'U2AF2', 'EPN1']
]



#for pretrained_model in pretrained_models:
#for pretrained_model in ["hyena_dna", "nucleotide_transformer"]:
for pretrained_model in ["enformer"]:
    df_path = model_data[pretrained_model]["df_path"]
    study_suffix = model_data[pretrained_model]["study_suffix"]
    study = f"{study_name}__{study_suffix}"
    os.makedirs(f'figures/{study}/neg_cor', exist_ok=True)
    df = pd.read_csv(df_path, sep="\t", index_col=[0])
    pred = np.load(f"prediction/{study}/prediction.npy")
    df2 = pd.DataFrame(pred)
    df2.columns = df.columns[1:]
    df2.index = df.index
    df1 = df.query('training == "test"').drop("training", axis=1)
    df2 = df2.loc[df1.index.to_list(), :]
    df1.index.name = "Real"
    df2.index.name = "Pred"
    for genes in genes_list:
        plot_scatter_real_pred(df1, df2, genes)
        correlation_heatmap(df1, df2, genes)




