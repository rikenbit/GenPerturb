# conda activate enformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os

study_names = [
"JialongJiang2024_B_cell_train",
"JialongJiang2024_CD4T_train",
"JialongJiang2024_CD8T_train",
"JialongJiang2024_Myeloid_train",
"JuliaJoung2023_TFAtlas_train",
"NormanWeissman2019_filtered_mixscape_exnp_train",
"ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
"ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
"ReplogleWeissman2022_rpe1_mixscape_exnp_train",
"Srivatsan2019_A549_train",
"Srivatsan2019_K562_train",
"Srivatsan2019_MCF7_train",
]


#pretrained_models = ["enformer", "hyena_dna_tss", "hyena_dna_last", "nucleotide_transformer_tss", "nucleotide_transformer_cls"]
pretrained_models = ["enformer"]
training = "enformer_transfer_epoch100_batch256_adamw5e3"

valid_pairs_dict = {}

for study_name in study_names:
    study = f'{study_name}__{training}'
    cor = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
    cor = cor.drop_duplicates(subset='Gene', keep=False)
    bed_file = f"fasta/{study_name}.bed"
    bed = pd.read_csv(bed_file, sep="\t",
        names=["chr_tss", "start_tss", "end_tss", "Gene", "score", "strand", "training"])
    bed = bed.drop_duplicates(subset='Gene', keep=False)
    cor = pd.merge(bed, cor, on=["Gene", "training"])
    grouped = bed.groupby('chr_tss')
    valid_pairs = []
    for name, group in grouped:
        merged = pd.merge(group, group, on='chr_tss', suffixes=('_1', '_2'))
        merged = merged[abs(merged['start_tss_1'] - merged['start_tss_2']) < 200000]
        valid_pairs.extend(
            merged[['Gene_1', 'Gene_2', 'training_1', 'chr_tss', 'start_tss_1', 'start_tss_2']].values.tolist()
        )
    base_valid_pairs_df = pd.DataFrame(valid_pairs, columns=['Gene1', 'Gene2', 'training_1', 'chr_tss', 'start_tss_1', 'start_tss_2'])
    cor_gene1 = cor.loc[:, ["Gene", "Correlation"]].rename(columns={"Gene": "Gene1", "Correlation": "Correlation_Gene1"})
    cor_gene2 = cor.loc[:, ["Gene", "Correlation"]].rename(columns={"Gene": "Gene2", "Correlation": "Correlation_Gene2"})
    for pretrained_model in pretrained_models:
        study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
        study = f"{study_name}__{study_suffix}"
        os.makedirs(f'figures/{study}/gene_density', exist_ok=True)
        df = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
        pred = np.load(f"prediction/{study}/prediction.npy")
        df1 = df.iloc[:, 1:]
        df2 = pd.DataFrame(pred)
        df2.columns = df.columns[1:]
        df2.index = df.index.copy()
        df1.index.name = "Real"
        df2.index.name = "Pred"
        df1_cor = df1.T.corr()
        df2_cor = df2.T.corr()
        valid_pairs_df_model = base_valid_pairs_df.copy()
        valid_pairs_df_model['correlation_real'] = valid_pairs_df_model.apply(
            lambda row: df1_cor.at[row['Gene1'], row['Gene2']] if row['Gene1'] in df1_cor.index and row['Gene2'] in df1_cor.columns else (
                df1_cor.at[row['Gene2'], row['Gene1']] if row['Gene2'] in df1_cor.index and row['Gene1'] in df1_cor.columns else None
            ), axis=1
        )
        valid_pairs_df_model['correlation_pred'] = valid_pairs_df_model.apply(
            lambda row: df2_cor.at[row['Gene1'], row['Gene2']] if row['Gene1'] in df2_cor.index and row['Gene2'] in df2_cor.columns else (
                df2_cor.at[row['Gene2'], row['Gene1']] if row['Gene2'] in df2_cor.index and row['Gene1'] in df2_cor.columns else None
            ), axis=1
        )
        valid_pairs_df_model = pd.merge(valid_pairs_df_model, cor_gene1, on="Gene1", how="left")
        valid_pairs_df_model = pd.merge(valid_pairs_df_model, cor_gene2, on="Gene2", how="left")
        valid_pairs_dict[(study_name, pretrained_model)] = valid_pairs_df_model


##### neighboring gene correlation
def plot_correlation(valid_pairs_df, stat='mean'):
    df = valid_pairs_df.copy()
    grouped = df.groupby(['Gene1', 'Correlation_Gene1', 'training_1'])['correlation_real'].agg([stat]).reset_index()
    plt.figure(figsize=(7/2.54, 6/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    colors = {'train': 'gray', 'val': 'steelblue', 'test': 'orengered'}
    order = ['train', 'val', 'test']
    grouped['training_1'] = pd.Categorical(grouped['training_1'], categories=order, ordered=True)
    sns.scatterplot(x=stat, y="Correlation_Gene1", data=grouped.sort_values('training_1'),
                    hue='training_1', palette=colors, s=10, edgecolor=None, legend=False)
    plt.xlabel(f'{stat} correlation with surrounding genes (real values)')
    plt.ylabel('Correlation between real and pred values')
    plt.tight_layout()
    os.makedirs(f'figures/{study}/surrounding_correlation', exist_ok=True)
    plt.savefig(f'figures/{study}/surrounding_correlation/surrounding_correlation_{dataset}_{stat}.svg')
    plt.clf()
    plt.close()


for study_name in study_names:
    for pretrained_model in pretrained_models:
        study = f"{study_name}__{study_suffix}"
        valid_pairs_df = valid_pairs_dict[(study_name, pretrained_model)]
        valid_pairs_df = valid_pairs_df[~(valid_pairs_df["Gene1"] == valid_pairs_df["Gene2"])]
        valid_pairs_df = valid_pairs_df.dropna()
        for stat in ["max", "min", "mean"]:
            plot_correlation(valid_pairs_df, stat=stat)

                                                     
