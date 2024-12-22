# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error


study_name = "JialongJiang2024_CD8T_train"
study_label = "Jialong et al. CD8T"

trainings = [
"enformer_transfer_epoch100_batch256_adamw5e3",
"enformer_transfer_epoch100_batch256_adamw5e3_fromdna",
"enformer_finetuning_epoch20_batch8_adamw5e3_lora_r16a2",
"enformer_finetuning_epoch20_batch8_adamw5e3_lora_r64a2",
"enformer_finetuning_epoch20_batch8_adamw5e3_lora_r256a2",
"enformer_finetuning_epoch20_batch8_adamw5e3_lora_r512a2",
"enformer_finetuning_epoch40_batch2_adamw5e3_full",
"enformer_finetuning_epoch40_batch2_adamw5e3_full_plr1e10",
"hyena_dna_last_transfer_epoch100_batch256_adamw5e3",
"hyena_dna_last_finetuning_epoch40_batch1_adamw5e3_full",
"hyena_dna_tss_transfer_epoch100_batch256_adamw5e3",
"hyena_dna_tss_finetuning_epoch40_batch1_adamw5e3_full",
"nucleotide_transformer_cls_transfer_epoch100_batch256_adamw5e3",
"nucleotide_transformer_cls_finetuning_epoch40_batch1_adamw5e3_full",
"nucleotide_transformer_tss_transfer_epoch100_batch256_adamw5e3",
"nucleotide_transformer_tss_finetuning_epoch40_batch1_adamw5e3_full",
]

pretrained_models = ["Enformer"] * 8 + ["HyenaDNA"] * 4 + ["Nucleotide Transformer"] * 4

xlabels = [
"Feature-based\nprediction from embedding",
"Feature-based\nprediction from DNA",
"Fine-tuning LoRA\nrank 16",
"Fine-tuning LoRA\nrank 64",
"Fine-tuning LoRA\nrank 256",
"Fine-tuning LoRA\nrank 512",
"Full fine-tuning\nlr1e4",
"Full fine-tuning\nlr1e10",
"Feature-based last\nprediction from embedding",
"Full fine-tuning last\nlr1e4",
"Feature-based TSS\nprediction from embedding",
"Full fine-tuning TSS\nlr1e4",
"Feature-based CLS\nprediction from embedding",
"Full fine-tuning CLS\nlr1e4",
"Feature-based TSS\nprediction from embedding",
"Full fine-tuning TSS\nlr1e4",
]

name_replace = {
'enformer': 'Enformer',
'hyena_dna': 'HyenaDNA',
'nucleotide_transformer': 'Nucleotide Transformer',
}



cor_acpertss  = pd.DataFrame()
cor_acgeness = pd.DataFrame()

for training, pretrained_model, xlabel in zip(trainings, pretrained_models, xlabels):
    study = f'{study_name}__{training}'
    cor_acperts = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
    cor_acgenes = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_genes.txt", sep="\t")
    cor_acperts["pretrained_model"] = pretrained_model
    cor_acgenes["pretrained_model"] = pretrained_model
    cor_acperts["training_method"] = xlabel
    cor_acgenes["training_method"] = xlabel
    cor_acpertss = pd.concat([cor_acpertss, cor_acperts], axis=0)
    cor_acgeness = pd.concat([cor_acgeness, cor_acgenes], axis=0)


cor_acpertss = cor_acpertss.query('training == "test"')
cor_acgeness = cor_acgeness.query('training == "test"')



def plot_boxplot_by_exp(cor, pretrained_model, palette, wide=10):
    fig = plt.figure(figsize=(wide/2.54, 7/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    ax = sns.boxplot(data=cor, x="training_method", y="Correlation", hue='Mean', width=0.6,
        fliersize=0, hue_order=["Very High", "High", "Medium", "Low", "Very Low"], palette=palette)
    ax.set_ylim(-0.6, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    plt.title(f'Correlation across perturbations\n{study_label}\n{pretrained_model}')
    plt.tight_layout()
    plt.savefig(f'across_study/compare_training/Correlation_across_perturbations_{pretrained_model.replace(" ", "_")}.svg')
    plt.clf()
    plt.close()


def plot_barplot(cor, pretrained_model, palette, wide=5):
    plt.figure(figsize=(wide/2.54, 8/2.54), dpi=300)
    plt.rcParams["font.size"] = 7
    ax = sns.barplot(data=cor, y="Correlation", x='training_method',
        palette=[palette] * len(set(cor["training_method"])))
    dot_size = 0.3
    ax.set_ylim(0, 1)
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Training methods', fontsize=6)
    plt.title(f'Correlation across genes\n{study_label}\n{pretrained_model}')
    plt.tight_layout()
    plt.savefig(f'across_study/compare_training/Correlation_across_genes_{pretrained_model.replace(" ", "_")}.svg')
    plt.clf()
    plt.close()




os.makedirs(f"across_study/compare_training", exist_ok=True)


for i,j,k in zip(["Enformer", "HyenaDNA", "Nucleotide Transformer"], ["Blues_r", "Oranges_r", "Greens_r"], [16,9,9]):
    plot_boxplot_by_exp(cor_acpertss.query('pretrained_model == @i'), i, j, k)


for i,j,k in zip(["Enformer", "HyenaDNA", "Nucleotide Transformer"], sns.color_palette("deep")[:3], [10,6.5,6.5]):
    plot_barplot(cor_acgeness.query('pretrained_model == @i'), i, j, k)




