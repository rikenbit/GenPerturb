# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_model_config import (
    studies,
    xlabels,
    pretrained_models as pretrained_models_raw,
    model_colors,
    name_replace,
    model_fixed_suffix,
)

cor_acpertss_dict = {}
cor_acgeness_dict = {}

for pretrained_model in pretrained_models_raw:
    cor_acpertss_dict[pretrained_model] = pd.DataFrame()
    cor_acgeness_dict[pretrained_model] = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
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
        cor_acperts["model"] = pretrained_model
        cor_acgenes["model"] = pretrained_model
        cor_acpertss_dict[pretrained_model] = pd.concat([cor_acpertss_dict[pretrained_model], cor_acperts], axis=0)
        cor_acgeness_dict[pretrained_model] = pd.concat([cor_acgeness_dict[pretrained_model], cor_acgenes], axis=0)


## merge stats across genes ##
os.makedirs("across_study/compare_models/merge_models", exist_ok=True)

def plot_barplot_by_exp(
    df,
    pretrained_models,
    stats="Correlation",
    output="_across_perturbations",
    yliml=0,
    ylimh=1,
    fig_width=None,
    color=None,
):
    if fig_width is None:
        fig_width = 15

    plt.figure(figsize=(fig_width/2.54, 6/2.54), dpi=300)
    EDGE_COLOR = "#B0B0B0"
    plt.rcParams.update({
        "font.size": 6,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

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
        title="Model",
    )
    plt.xticks(rotation=60, ha="right", rotation_mode="anchor")
    plt.title(f"{stats} across genes")
    plt.tight_layout()
    plt.savefig(f"across_study/compare_models/merge_models/{stats}_{output}.svg")
    plt.clf()
    plt.close()


cor_acpertss_merge = pd.concat(
    [cor_acpertss_dict[i] for i in pretrained_models_raw],
    ignore_index=True,
)
cor_acgeness_merge = pd.concat(
    [cor_acgeness_dict[i] for i in pretrained_models_raw],
    ignore_index=True,
)

cor_acpertss_merge["model"] = cor_acpertss_merge["model"].replace(name_replace)
cor_acgeness_merge["model"] = cor_acgeness_merge["model"].replace(name_replace)

pretrained_models = [name_replace.get(i, i) for i in pretrained_models_raw]

if model_colors is not None and len(model_colors) < len(pretrained_models):
    raise ValueError(
        f"Palette length ({len(model_colors)}) is smaller than number of models ({len(pretrained_models)})."
    )

plot_barplot_by_exp(
    cor_acgeness_merge.query('training == "test"'),
    pretrained_models,
    stats="Correlation",
    output="across_genes",
    fig_width=13,
    color=model_colors,
)
