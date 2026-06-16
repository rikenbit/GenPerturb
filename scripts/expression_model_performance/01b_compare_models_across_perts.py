# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_model_config import (
    studies,
    xlabels,
    pretrained_models,
    cmaps,
    model_colors,
    name_replace,
    model_fixed_suffix,
)


EDGE_COLOR = "#B0B0B0"      # soft gray for spines and tick marks
BOX_LINE_COLOR = "#666666"   # darker gray for box/whisker/percentile lines
FONT_SIZE = 7
SOFT_RC = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.edgecolor": EDGE_COLOR,
    "axes.linewidth": 0.6,
    "xtick.color": EDGE_COLOR,
    "ytick.color": EDGE_COLOR,
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def plot_boxplot_by_exp(cor, output="", outdir="tmp", yliml=0, pretrained_model="tmp",
                         cmap="Greys_r", model_color="#6C757D", hue=False):
    if hue:
        plt.figure(figsize=(14/2.54, 7.5/2.54), dpi=300)
        plt.rcParams.update(SOFT_RC)
        ax = sns.boxplot(
            data=cor, x="study", y="Correlation", hue='Mean', width=0.8, fliersize=0,
            hue_order=["Very High", "High", "Medium", "Low", "Very Low"],
            palette=cmap,
            linewidth=0.7,
            boxprops={"edgecolor": BOX_LINE_COLOR},
            whiskerprops={"color": BOX_LINE_COLOR},
            capprops={"color": BOX_LINE_COLOR},
            medianprops={"color": BOX_LINE_COLOR},
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
    else:
        plt.figure(figsize=(9/2.54, 8/2.54), dpi=300)
        plt.rcParams.update(SOFT_RC)
        ax = sns.barplot(
            data=cor, x="study", y="Correlation",
            color=model_color,
            saturation=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )
    ax.set_ylim(min(yliml, 0), 1)
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Datasets')
    plt.title(f'{pretrained_model}\nCorrelation{output.replace("_", " ")}', fontsize=FONT_SIZE)
    plt.tight_layout()
    pretrained_model = pretrained_model.replace(" ", "_")
    plt.savefig(f'across_study/compare_models/correlation_across_perturbations/{pretrained_model}/{outdir}/{pretrained_model}_Correlation{output}.svg')
    plt.clf()
    plt.close()





for pretrained_model, cmap, model_color in zip(pretrained_models, cmaps, model_colors):
    cor_acpertss  = pd.DataFrame()
    cor_acgeness = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
        if study_name.startswith("Wu2024"):
            continue
        suffix = model_fixed_suffix.get(pretrained_model, "transfer_epoch100_batch256_adamw5e3")
        study = f'{study_name}__{pretrained_model}_{suffix}'
        perts_path = f"figures/{study}/cor_matrix/correlation_across_perts.txt"
        genes_path = f"figures/{study}/cor_matrix/correlation_across_genes.txt"
        if not (os.path.exists(perts_path) and os.path.exists(genes_path)):
            print(f"[skip] missing files for {study}")
            continue
        cor_acperts = pd.read_csv(perts_path, sep="\t")
        cor_acgenes = pd.read_csv(genes_path, sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acpertss     = pd.concat([cor_acpertss, cor_acperts], axis=0)
        cor_acgeness    = pd.concat([cor_acgeness, cor_acgenes], axis=0)
    outdir="adamw5e3"
    pretrained_model = name_replace.get(pretrained_model, pretrained_model)
    cor_acpertss = cor_acpertss.reset_index(drop=True)
    cor_acgeness = cor_acgeness.reset_index(drop=True)
    os.makedirs(f'across_study/compare_models/correlation_across_perturbations/{pretrained_model.replace(" ", "_")}/{outdir}', exist_ok=True)
    for training in ["train", "val", "test", "all"]:
        if training == "all":
            cor_acperts_subset = cor_acpertss
            cor_acgenes_subset = cor_acgeness
        else:
            cor_acperts_subset = cor_acpertss.query('training == @training')
            cor_acgenes_subset = cor_acgeness.query('training == @training')
        plot_boxplot_by_exp(cor_acperts_subset,
            yliml=cor_acperts_subset["Correlation"].min(),
            output=f"_across_perturbations_({training})", outdir=outdir,
            pretrained_model=pretrained_model, cmap=cmap, model_color=model_color, hue=True)
        plot_boxplot_by_exp(cor_acgenes_subset,
            yliml=cor_acgenes_subset["Correlation"].min(),
            output=f"_across_genes_({training})", outdir=outdir,
            pretrained_model=pretrained_model, cmap=cmap, model_color=model_color, hue=False)


