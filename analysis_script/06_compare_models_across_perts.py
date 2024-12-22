# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error

studies = [
"NormanWeissman2019_filtered_mixscape_exnp_train",
"ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
"ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
"ReplogleWeissman2022_rpe1_mixscape_exnp_train",
"JialongJiang2024_Myeloid_train",
"JialongJiang2024_CD4T_train",
"JialongJiang2024_CD8T_train",
"JialongJiang2024_B_cell_train",
"JuliaJoung2023_TFAtlas_train",
"Srivatsan2019_A549_train",
"Srivatsan2019_K562_train",
"Srivatsan2019_MCF7_train",
]

xlabels = [
"Norman et al.",
"Replogle et al. essential",
"Replogle et al. gwps",
"Replogle et al. rpe1",
"Jialong et al. myeloid",
"Jialong et al. CD4T",
"Jialong et al. CD8T",
"Jialong et al. Bcell",
"Julia et al.",
"Srivatsan et al. A549",
"Srivatsan et al. K562",
"Srivatsan et al. MCF7",
]

name_replace = {
'enformer': 'Enformer',
'hyena_dna_tss': 'HyenaDNA TSS',
'hyena_dna_last': 'HyenaDNA last',
'nucleotide_transformer_tss': 'Nucleotide Transformer TSS',
'nucleotide_transformer_cls': 'Nucleotide Transformer CLS',
}



def plot_boxplot_by_exp(cor, output="", outdir="tmp", yliml=0, pretrained_model="tmp", hue=False):
    if hue:
        plt.figure(figsize=(13/2.54, 7/2.54), dpi=300)
        plt.rcParams["font.size"] = 7
        ax = sns.boxplot(data=cor, x="study", y="Correlation", hue='Mean', width=0.8, fliersize=0,
            hue_order=["Very High", "High", "Medium", "Low", "Very Low"], palette=cmap)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
    else:
        plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rcParams["font.size"] = 7
        ax = sns.barplot(data=cor, x="study", y="Correlation", palette="mako") ## future deprecated
    ax.set_ylim(min(yliml, 0), 1)
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Datasets')
    plt.title(f'{pretrained_model}\nCorrelation{output.replace("_", " ")}')
    plt.tight_layout()
    pretrained_model = pretrained_model.replace(" ", "_")
    plt.savefig(f'across_study/compare_models/correlation_across_perturbations/{pretrained_model}/{outdir}/{pretrained_model}_Correlation{output}.svg')
    plt.clf()
    plt.close()


pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
]

cmaps = [
"Blues_r",
"Oranges_r",
"Oranges_r",
"Greens_r",
"Greens_r",
]


epoch = 100
batch = 256


for pretrained_model, cmap in zip(pretrained_models, cmaps):
    cor_acpertss  = pd.DataFrame()
    cor_acgeness = pd.DataFrame()
    cor_acpertss_fc  = pd.DataFrame()
    cor_acgeness_fc = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
        study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
        cor_acperts = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
        cor_acgenes = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_genes.txt", sep="\t")
        cor_acperts_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation_across_perts.txt", sep="\t")
        cor_acgenes_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation_across_genes.txt", sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acperts_fc["study"] = xlabel
        cor_acgenes_fc["study"] = xlabel
        cor_acpertss     = pd.concat([cor_acpertss, cor_acperts], axis=0)
        cor_acgeness    = pd.concat([cor_acgeness, cor_acgenes], axis=0)
        cor_acpertss_fc  = pd.concat([cor_acpertss_fc, cor_acperts_fc], axis=0)
        cor_acgeness_fc = pd.concat([cor_acgeness_fc, cor_acgenes_fc], axis=0)
    outdir="adamw5e3"
    pretrained_model = name_replace.get(pretrained_model, pretrained_model)
    os.makedirs(f'across_study/compare_models/correlation_across_perturbations/{pretrained_model.replace(" ", "_")}/{outdir}', exist_ok=True)
    for training in ["train", "val", "test"]:
        plot_boxplot_by_exp(cor_acpertss.query('training == @training'),
            yliml=cor_acpertss.query('training == @training')["Correlation"].min(),
            output=f"_across_perturbations_({training})", outdir=outdir,
            pretrained_model=pretrained_model, hue=True)
        plot_boxplot_by_exp(cor_acgeness.query('training == @training'),
            yliml=cor_acgeness.query('training == @training')["Correlation"].min(),
            output=f"_across_genes_({training})", outdir=outdir,
            pretrained_model=pretrained_model, hue=False)
        plot_boxplot_by_exp(cor_acpertss_fc.query('training == @training'),
            yliml=cor_acpertss_fc.query('training == @training')["Correlation"].min(),
            output=f"_across_perturbations_({training}_FC)", outdir=outdir,
            pretrained_model=pretrained_model, hue=True)
        plot_boxplot_by_exp(cor_acgeness_fc.query('training == @training'),
            yliml=cor_acgeness_fc.query('training == @training')["Correlation"].min(),
            output=f"_across_genes_({training}_FC)", outdir=outdir,
            pretrained_model=pretrained_model, hue=False)


