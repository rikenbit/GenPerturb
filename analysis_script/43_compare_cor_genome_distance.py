#conda activate enformer
import pandas as pd
import numpy as np
from pybedtools import BedTool
import seaborn as sns
import matplotlib.pyplot as plt
import os


study_names = [
"NormanWeissman2019_filtered_mixscape_exnp_train",
'NormanWeissman2019_filtered_mixscape_exnp_train_shuffle_evenly0',
'NormanWeissman2019_filtered_mixscape_exnp_train_shuffle_evenly1',
'NormanWeissman2019_filtered_mixscape_exnp_train_shuffle_evenly2',
'NormanWeissman2019_filtered_mixscape_exnp_train_shuffle_evenly3',
'NormanWeissman2019_filtered_mixscape_exnp_train_shuffle_evenly4',
"JialongJiang2024_CD8T_train",
'JialongJiang2024_CD8T_train_shuffle_evenly0',
'JialongJiang2024_CD8T_train_shuffle_evenly1',
'JialongJiang2024_CD8T_train_shuffle_evenly2',
'JialongJiang2024_CD8T_train_shuffle_evenly3',
'JialongJiang2024_CD8T_train_shuffle_evenly4',
]

pretrained_models = [
    "enformer",
    "hyena_dna_tss",
    "nucleotide_transformer_tss"
]

df2s = {}

for study_name in study_names:
    df = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
    df_index = df.index
    df_columns = df.columns[1:]
    df = df.query('training == "test"').drop("training", axis=1)
    bed = pd.read_csv(f'fasta/{study_name}.bed', sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
    bed.index = bed["Gene"]
    df2s[study_name] = {}
    for pretrained_model in pretrained_models:
        try:
            study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
            study = f'{study_name}__{study_suffix}'
            pred = np.load(f"prediction/{study}/prediction.npy")
            df2 = pd.DataFrame(pred)
            df2.columns = df_columns
            df2.index = df_index
            df2 = df2.loc[df.index.to_list(), :]
            df2s[study_name][pretrained_model] = df2
        except:
            continue


def plot_violinplot(data, x="", y="Correlation", hue="", title="", study_name=""):
    model_num = len(set(data["dataset"]))
    pallete = sns.color_palette("Dark2")
    plt.figure(figsize=((10 + model_num*4)/2.54, 12/2.54), dpi=300)
    plt.rcParams["font.size"] = 7
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=data, hue=hue, x=x, y=y, palette=pallete, cut=0, fill=False, density_norm="width", 
        order=[i for i in ["Real", "Enformer", "HyenaDNA TSS", "Nucleotide Transformer TSS"] if i in data["dataset"].values])
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), )
    plt.tight_layout()
    os.makedirs(f'across_study/compare_models/correlation_genome_distance/{study_name}', exist_ok=True)
    plt.savefig(f"across_study/compare_models/correlation_genome_distance/{study_name}/correlation_{title}.svg")


def cal_correlation(data, study, value="pred"):
    correlation_matrix = data.T.corr()
    correlation_matrix.index.name = ""
    correlation_matrix_stack = correlation_matrix.stack().reset_index()
    correlation_matrix_stack.columns = ["gene1", "gene2", "Correlation"]
    dist_summary = pd.DataFrame()
    for chrom, group in bed.groupby("chr"):
        distances = group["start"].values - group["start"].values[:, None]
        dist_df = pd.DataFrame(distances, index=group["Gene"], columns=group["Gene"])
        dist_df.index.name = ""
        dist_df = dist_df.stack().reset_index()
        dist_df.columns = ["gene1", "gene2", "bp"]
        dist_df = dist_df.query('bp >= 0')
        dist_summary = pd.concat([dist_summary, dist_df], axis=0)
    res = pd.merge(correlation_matrix_stack, dist_summary, on=["gene1", "gene2"], how="left")
    res = res.dropna()
    res = res[~(res["gene1"] == res["gene2"])]
    os.makedirs(f'figures/{study}/TAD', exist_ok=True)
    res = res.sort_values("bp")
    bins = [-1, 5000, 10000, 50000, 100000, 150000, 200000, float('inf')]
    labels = [f"{int(bins[i])+1}-{int(bins[i+1])} bp" if bins[i+1] != float('inf') else f">{int(bins[i])} bp" for i in range(len(bins)-1)]
    res['distance'] = pd.cut(res['bp'], bins=bins, labels=labels, right=False)
    res.to_csv(f"figures/{study}/TAD/correlation_gene_dist_{value}.bed", sep="\t", index=False)
    return res

pretrained_models = [
    "enformer",
#    "hyena_dna_tss",
#    "nucleotide_transformer_tss"
]


name_replace = {
'enformer': 'Enformer',
'hyena_dna_tss': 'HyenaDNA TSS',
'hyena_dna_last': 'HyenaDNA last',
'nucleotide_transformer_tss': 'Nucleotide Transformer TSS',
'nucleotide_transformer_cls': 'Nucleotide Transformer CLS',
}



for study_name in study_names:
    real_dict = {}
    preds_dict = {}
    df = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
    df_index = df.index
    df_columns = df.columns[1:]
    df = df.query('training == "test"').drop("training", axis=1)
    study_suffix = f"enformer_transfer_epoch100_batch256_adamw5e3"
    study_full_name = f'{study_name}__{study_suffix}'
    real = cal_correlation(df, study_full_name, value="real")
    real["dataset"] = "Real"
    real["study_name"] = study_name
    real_dict[study_name] = real
    preds = {}
    for pretrained_model in pretrained_models:
        try:
            study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
            study_full_name = f'{study_name}__{study_suffix}'
            pred = cal_correlation(df2s[study_name][pretrained_model], study_full_name, value="pred")
            pred["dataset"] = name_replace.get(pretrained_model, pretrained_model)
            pred["study_name"] = study_name
            preds[pretrained_model] = pred
        except:
            continue
    preds_dict[study_name] = preds
    real = real_dict[study_name]
    preds = preds_dict[study_name]
    combined = pd.concat([real] + [preds[model] for model in pretrained_models if model in preds], ignore_index=True)
    plot_violinplot(combined, x="dataset", y="Correlation", hue="distance", title="_".join(pretrained_models), study_name=study_name)


