#conda activate enformer
import pandas as pd
import numpy as np
from pybedtools import BedTool
import seaborn as sns
import matplotlib.pyplot as plt
import os


study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman_K562_CRISPRa"


#pretrained_models = ["enformer", "hyena_dna", "nucleotide_transformer"]
pretrained_model = "enformer"
study_suffixes = [
    f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3",
    f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3_masked",
]

lengths = [
    "196,608 bp",
    "10,000 bp",
]


data = pd.DataFrame()
for study_suffix, length in zip(study_suffixes, lengths):
    study = f"{study_name}__{study_suffix}"
    cor = pd.read_csv(f"figures/{study}/TAD/correlation_gene_dist_pred.bed", sep="\t")
    cor["input_length"] = length 
    data = pd.concat([data, cor])


def plot_violinplot(data, x="", y="Correlation", hue="", title=""):
    pallete = sns.color_palette("Dark2")
    plt.figure(figsize=(18/2.54, 12/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=data, hue=hue, x=x, y=y, palette=pallete, cut=0, fill=False,
        density_norm="width", order=data[x].drop_duplicates().to_list())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    os.makedirs(f'across_study/compare_models/correlation_genome_distance', exist_ok=True)
    plt.savefig(f"across_study/compare_models/correlation_genome_distance/correlation_{title}.svg")


plot_violinplot(data, x="distance", y="Correlation", hue="input_length", title="genome_distance_mask")


def plot_boxplot_by_exp(cor):
    plt.figure(figsize=(12/2.54, 8/2.54), dpi=300)
    ax = sns.boxplot(data=cor, x="input_length", y="Correlation",
        hue='Mean', width=0.6, fliersize=0.2,
        hue_order=["Very High", "High", "Medium", "Low", "Very Low"], palette="Blues_r")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
    plt.rcParams["font.size"] = 6
    dot_size = 0.3
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(f'Correlation (input mask)')
    plt.tight_layout()
    plt.savefig(f"across_study/compare_models/correlation_genome_distance/correlation_mask_across_perturbations.svg")
    plt.clf()
    plt.close()


def plot_barplot(cor):
    plt.figure(figsize=(4/2.54, 8/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    ax = sns.barplot(data=cor, y="Correlation", x='input_length', palette="Dark2")
    dot_size = 0.3
    ax.set_ylim(0, 1)
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Input length')
    plt.title(f'Input mask')
    plt.tight_layout()
    plt.savefig(f"across_study/compare_models/correlation_genome_distance/correlation_mask_across_genes.svg")
    plt.clf()
    plt.close()


cor_seqs  = pd.DataFrame()
cor_perts = pd.DataFrame()

for study_suffix, length in zip(study_suffixes, lengths):
    study = f'{study_name}__{study_suffix}'
    cor_seq = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
    cor_pert = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_genes.txt", sep="\t")
    cor_seq["input_length"] = length
    cor_pert["input_length"] = length
    cor_seqs     = pd.concat([cor_seqs, cor_seq], axis=0)
    cor_perts    = pd.concat([cor_perts, cor_pert], axis=0)


plot_boxplot_by_exp(cor_seqs.query('training == "test"'))
plot_barplot(cor_perts.query('training == "test"'))


