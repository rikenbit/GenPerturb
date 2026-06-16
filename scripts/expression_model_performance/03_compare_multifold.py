# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_model_config import name_replace

# Base datasets to compare across AlphaGenome folds
datasets = [
"NormanWeissman2019_filtered_mixscape_exnp",
#"ReplogleWeissman2022_K562_essential_mixscape_exnp",
#"ReplogleWeissman2022_K562_gwps_mixscape_exnp",
#"ReplogleWeissman2022_rpe1_mixscape_exnp",
#"JialongJiang2024_Myeloid",
#"JialongJiang2024_CD4T",
"JialongJiang2024_CD8T",
#"JialongJiang2024_B_cell",
#"Srivatsan2019_A549",
#"Srivatsan2019_K562",
#"Srivatsan2019_MCF7",
]

fold_models = [
"alphagenome",
"alphagenome_fold_0",
"alphagenome_fold_1",
"alphagenome_fold_2",
"alphagenome_fold_3",
]

xlabels = [
"All Folds",
"Fold 0",
"Fold 1",
"Fold 2",
"Fold 3",
]

cmap = "YlOrBr_r"
single_color = "#BC5765"

epoch = 100
batch = 256

EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"
_SOFT_RC = {
    "axes.edgecolor": EDGE_COLOR,
    "axes.linewidth": 0.6,
    "xtick.color": EDGE_COLOR,
    "ytick.color": EDGE_COLOR,
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def plot_boxplot_by_exp(cor, output="", outdir="tmp", yliml=0, pretrained_model="tmp", dataset="tmp",
                         cmap="YlOrBr_r", single_color="#BC5765", title_name=None,
                         file_name=None, hue=False):
    if hue:
        plt.figure(figsize=(13/2.54, 7/2.54), dpi=300)
        plt.rcParams.update({"font.size": 7, **_SOFT_RC})
        ax = sns.boxplot(
            data=cor, x="study", y="Correlation", hue='Mean', width=0.8, fliersize=0,
            hue_order=["Very High", "High", "Medium", "Low", "Very Low"], palette=cmap,
            linewidth=0.7,
            boxprops={"edgecolor": LINE_COLOR},
            whiskerprops={"color": LINE_COLOR},
            capprops={"color": LINE_COLOR},
            medianprops={"color": LINE_COLOR},
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
    else:
        plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rcParams.update({"font.size": 7, **_SOFT_RC})
        ax = sns.barplot(
            data=cor, x="study", y="Correlation",
            color=single_color,
            saturation=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
        )
    ax.set_ylim(min(yliml, 0), 1)
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Datasets')
    title_name = title_name if title_name is not None else pretrained_model
    plt.title(f'{title_name}\nCorrelation{output.replace("_", " ")}')
    plt.tight_layout()
    file_name = file_name if file_name is not None else pretrained_model
    pretrained_model_fname = file_name.replace(" ", "_")
    plt.savefig(f'across_study/multifold/correlation_across_perturbations/{dataset}/{pretrained_model_fname}/{outdir}/{pretrained_model_fname}_Correlation{output}.svg')
    plt.clf()
    plt.close()


for dataset in datasets:
    study_name = f"{dataset}_train"
    cor_acpertss = pd.DataFrame()
    cor_acgeness = pd.DataFrame()
    for fold_model, xlabel in zip(fold_models, xlabels):
        study = f'{study_name}__{fold_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
        cor_acperts = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
        cor_acgenes = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_genes.txt", sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acpertss = pd.concat([cor_acpertss, cor_acperts], axis=0)
        cor_acgeness = pd.concat([cor_acgeness, cor_acgenes], axis=0)
    outdir = "adamw5e3"
    pretrained_model = "AlphaGenome"
    display_name = "GenPerturb (AlphaGenome transfer)"
    cor_acpertss = cor_acpertss.reset_index(drop=True)
    cor_acgeness = cor_acgeness.reset_index(drop=True)
    os.makedirs(f'across_study/multifold/correlation_across_perturbations/{dataset}/{pretrained_model}/{outdir}', exist_ok=True)
    for training in ["train", "val", "test"]:
        plot_boxplot_by_exp(cor_acpertss.query('training == @training'),
            yliml=cor_acpertss.query('training == @training')["Correlation"].min(),
            output=f"_across_perturbations_({training})", outdir=outdir,
            pretrained_model=pretrained_model, title_name=display_name, file_name=pretrained_model,
            dataset=dataset, cmap=cmap, single_color=single_color, hue=True)
        plot_boxplot_by_exp(cor_acgeness.query('training == @training'),
            yliml=cor_acgeness.query('training == @training')["Correlation"].min(),
            output=f"_across_genes_({training})", outdir=outdir,
            pretrained_model=pretrained_model, title_name=display_name, file_name=pretrained_model,
            dataset=dataset, cmap=cmap, single_color=single_color, hue=False)
