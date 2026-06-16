import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_model_config import (
    studies,
    xlabels,
    name_replace,
    model_fixed_suffix,
)

name_replace.update({
    "baseline_peturbmean": "Baseline (Perturbation Mean)",
    "baseline_shuffle": "Baseline (Shuffle)",
})

baseline_models = [
    "baseline_control",
    "baseline_peturbmean",
    "baseline_shuffle",
]

baseline_cmaps = [
    "Greys_r",
    "Greys_r",
    "Greys_r",
]

baseline_palette = ["#2d2d2d", "#7a7a7a", "#b5b5b5"]
baseline_single_colors = {
    "baseline_control": "#2d2d2d",
    "baseline_peturbmean": "#7a7a7a",
    "baseline_shuffle": "#b5b5b5",
}
baseline_fig_width = 11

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



def plot_boxplot_by_exp(cor, output="", outdir="tmp", yliml=0, pretrained_model="tmp",
                         cmap="Greys_r", single_color="#666666", hue=False):
    if hue:
        plt.figure(figsize=(14/2.54, 7/2.54), dpi=300)
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
        plt.figure(figsize=(9/2.54, 8/2.54), dpi=300)
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
    plt.title(f'{pretrained_model}\nCorrelation{output.replace("_", " ")}')
    plt.tight_layout()
    pretrained_model = pretrained_model.replace(" ", "_")
    plt.savefig(f'across_study/compare_models/correlation_across_perturbations/{pretrained_model}/{outdir}/{pretrained_model}_Correlation{output}.svg')
    plt.clf()
    plt.close()


for pretrained_model, cmap in zip(baseline_models, baseline_cmaps):
    cor_acpertss = pd.DataFrame()
    cor_acgeness = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
        suffix = model_fixed_suffix.get(pretrained_model, "transfer_epoch100_batch256_adamw5e3")
        study = f'{study_name}__{pretrained_model}_{suffix}'
        perts_file = f"figures/{study}/cor_matrix/correlation_across_perts.txt"
        genes_file = f"figures/{study}/cor_matrix/correlation_across_genes.txt"
        if not os.path.exists(perts_file) or not os.path.exists(genes_file):
            print(f"  WARNING: Missing files for {study}, skipping")
            continue
        cor_acperts = pd.read_csv(perts_file, sep="\t")
        cor_acgenes = pd.read_csv(genes_file, sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acpertss = pd.concat([cor_acpertss, cor_acperts], axis=0)
        cor_acgeness = pd.concat([cor_acgeness, cor_acgenes], axis=0)
    outdir = "adamw5e3"
    model_display = name_replace.get(pretrained_model, pretrained_model)
    single_color = baseline_single_colors.get(pretrained_model, "#666666")
    cor_acpertss = cor_acpertss.reset_index(drop=True)
    cor_acgeness = cor_acgeness.reset_index(drop=True)
    os.makedirs(f'across_study/compare_models/correlation_across_perturbations/{model_display.replace(" ", "_")}/{outdir}', exist_ok=True)
    for training in ["train", "val", "test"]:
        plot_boxplot_by_exp(cor_acpertss.query('training == @training'),
            yliml=cor_acpertss.query('training == @training')["Correlation"].min(),
            output=f"_across_perturbations_({training})", outdir=outdir,
            pretrained_model=model_display, cmap=cmap, single_color=single_color, hue=True)
        plot_boxplot_by_exp(cor_acgeness.query('training == @training'),
            yliml=cor_acgeness.query('training == @training')["Correlation"].min(),
            output=f"_across_genes_({training})", outdir=outdir,
            pretrained_model=model_display, cmap=cmap, single_color=single_color, hue=False)



cor_acpertss_dict = {}
cor_acgeness_dict = {}

for pretrained_model in baseline_models:
    cor_acpertss_dict[pretrained_model] = pd.DataFrame()
    cor_acgeness_dict[pretrained_model] = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
        suffix = model_fixed_suffix.get(pretrained_model, "transfer_epoch100_batch256_adamw5e3")
        study = f'{study_name}__{pretrained_model}_{suffix}'
        perts_file = f"figures/{study}/cor_matrix/correlation_across_perts.txt"
        genes_file = f"figures/{study}/cor_matrix/correlation_across_genes.txt"
        if not os.path.exists(perts_file) or not os.path.exists(genes_file):
            continue
        cor_acperts = pd.read_csv(perts_file, sep="\t")
        cor_acgenes = pd.read_csv(genes_file, sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acperts["model"] = pretrained_model
        cor_acgenes["model"] = pretrained_model
        cor_acpertss_dict[pretrained_model] = pd.concat([cor_acpertss_dict[pretrained_model], cor_acperts], axis=0)
        cor_acgeness_dict[pretrained_model] = pd.concat([cor_acgeness_dict[pretrained_model], cor_acgenes], axis=0)

os.makedirs("across_study/compare_models/merge_models", exist_ok=True)


def plot_barplot_by_exp(
    df,
    pretrained_models,
    stats="Correlation",
    value="",
    output="_across_perturbations",
    yliml=0,
    ylimh=1,
    fig_width=None,
    color=None,
):
    if fig_width is None:
        fig_width = 15

    plt.figure(figsize=(fig_width/2.54, 6/2.54), dpi=300)
    plt.rcParams.update({"font.size": 6, **_SOFT_RC})

    ax = sns.barplot(
        data=df,
        x="study",
        y=stats,
        hue="model",
        hue_order=pretrained_models,
        palette=color,
        saturation=1.0,
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
    )
    ax.set_ylim(yliml, ylimh)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Baseline model",
    )
    plt.xticks(rotation=60, ha="right", rotation_mode="anchor")
    plt.title(f"{stats} across genes")
    plt.tight_layout()
    plt.savefig(f"across_study/compare_models/merge_models/{stats}_{output}_{value}.svg")
    plt.clf()
    plt.close()


cor_acpertss_merge = pd.concat(
    [cor_acpertss_dict[i] for i in baseline_models],
    ignore_index=True,
)
cor_acgeness_merge = pd.concat(
    [cor_acgeness_dict[i] for i in baseline_models],
    ignore_index=True,
)

cor_acpertss_merge["model"] = cor_acpertss_merge["model"].replace(name_replace)
cor_acgeness_merge["model"] = cor_acgeness_merge["model"].replace(name_replace)

baseline_display_names = [name_replace.get(i, i) for i in baseline_models]

plot_barplot_by_exp(
    cor_acgeness_merge.query('training == "test"'),
    baseline_display_names,
    stats="Correlation",
    output="across_genes",
    value="baseline",
    fig_width=baseline_fig_width,
    color=baseline_palette,
)
