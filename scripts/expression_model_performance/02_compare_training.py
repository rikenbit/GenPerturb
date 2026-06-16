# conda activate alphagenome
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_model_config import name_replace


study_configs = [
    {"name": "NormanWeissman2019_filtered_mixscape_exnp_train", "label": "Norman et al."},
    {"name": "JialongJiang2024_CD8T_train", "label": "Jialong et al. CD8T"},
]

model_configs = [
    {
        "name": "Enformer",
        "display_name": "GenPerturb (Enformer transfer)",
        "cmap": "Blues_r",
        "bar_color": sns.color_palette("deep")[0],
        "box_width": 7,
        "bar_width": 5,
        "trainings": [
            ("enformer_transfer_epoch100_batch256_adamw5e3", "Feature-based\nprediction from embedding"),
        ],
    },
    {
        "name": "Borzoi",
        "display_name": "GenPerturb (Borzoi transfer)",
        "cmap": "Purples_r",
        "bar_color": sns.color_palette("deep")[4],
        "box_width": 7,
        "bar_width": 5,
        "trainings": [
            ("borzoi_transfer_epoch100_batch256_adamw5e3", "Feature-based\nprediction from embedding"),
        ],
    },
    {
        "name": "AlphaGenome",
        "display_name": "GenPerturb (AlphaGenome transfer)",
        "cmap": "YlOrBr_r",
        "bar_color": sns.color_palette("deep")[5],
        "box_width": 14,
        "bar_width": 9,
        "trainings": [
            ("alphagenome_transfer_epoch100_batch256_adamw5e3", "Feature-based\nprediction from embedding"),
            ("alphagenome_finetuning_epoch100_batch2_adamw5e3_lora_r64a2", "Fine-tuning LoRA\nrank 64"),
            ("alphagenome_finetuning_epoch100_batch2_adamw5e3_lora_r256a2", "Fine-tuning LoRA\nrank 256"),
            ("alphagenome_finetuning_epoch100_batch2_adamw5e3_lora_r512a2", "Fine-tuning LoRA\nrank 512"),
            ("alphagenome_finetuning_epoch150_batch2_adamw5e3_full", "Full fine-tuning"),
        ],
    },
]


EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"


def load_correlations(study_name, model_config):
    cor_acperts_list = []
    cor_acgenes_list = []
    model_name = model_config["name"]

    for training, xlabel in model_config["trainings"]:
        study = f"{study_name}__{training}"
        perts_file = f"figures/{study}/cor_matrix/correlation_across_perts.txt"
        genes_file = f"figures/{study}/cor_matrix/correlation_across_genes.txt"

        if not os.path.exists(perts_file) or not os.path.exists(genes_file):
            print(f"  WARNING: Missing files for {study}, skipping")
            continue

        cor_acperts = pd.read_csv(perts_file, sep="\t")
        cor_acgenes = pd.read_csv(genes_file, sep="\t")
        cor_acperts["pretrained_model"] = model_name
        cor_acgenes["pretrained_model"] = model_name
        cor_acperts["training_method"] = xlabel
        cor_acgenes["training_method"] = xlabel
        cor_acperts_list.append(cor_acperts)
        cor_acgenes_list.append(cor_acgenes)

    if not cor_acperts_list:
        return None, None

    cor_acperts = pd.concat(cor_acperts_list, axis=0).query('training == "test"').reset_index(drop=True)
    cor_acgenes = pd.concat(cor_acgenes_list, axis=0).query('training == "test"').reset_index(drop=True)
    return cor_acperts, cor_acgenes


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


def plot_boxplot_by_exp(cor, model_name, display_name, study_label, palette, wide=10):
    fig = plt.figure(figsize=(wide / 2.54, 7 / 2.54), dpi=300)
    plt.rcParams.update({"font.size": 6, **_SOFT_RC})
    ax = sns.boxplot(
        data=cor, x="training_method", y="Correlation", hue="Mean", width=0.6,
        fliersize=0, hue_order=["Very High", "High", "Medium", "Low", "Very Low"],
        palette=palette,
        linewidth=0.7,
        boxprops={"edgecolor": LINE_COLOR},
        whiskerprops={"color": LINE_COLOR},
        capprops={"color": LINE_COLOR},
        medianprops={"color": LINE_COLOR},
    )
    ax.set_ylim(-0.6, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., title="Expression")
    plt.xticks(rotation=60, ha="right", rotation_mode="anchor")
    plt.title(f"Correlation across perturbations\n{study_label}\n{display_name}")
    plt.tight_layout()
    model_file = model_name.replace(" ", "_")
    study_file = study_label.replace(" ", "_").replace(".", "")
    plt.savefig(f"across_study/compare_training/Correlation_across_perturbations_{study_file}_{model_file}.svg")
    plt.clf()
    plt.close()


def plot_barplot(cor, model_name, display_name, study_label, color, wide=5):
    plt.figure(figsize=(wide / 2.54, 8 / 2.54), dpi=300)
    plt.rcParams.update({"font.size": 7, **_SOFT_RC})
    ax = sns.barplot(
        data=cor, y="Correlation", x="training_method",
        palette=[color] * len(set(cor["training_method"])),
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
    )
    ax.set_ylim(0, 1)
    plt.xticks(rotation=60, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Training methods", fontsize=6)
    plt.title(f"Correlation across genes\n{study_label}\n{display_name}")
    plt.tight_layout()
    model_file = model_name.replace(" ", "_")
    study_file = study_label.replace(" ", "_").replace(".", "")
    plt.savefig(f"across_study/compare_training/Correlation_across_genes_{study_file}_{model_file}.svg")
    plt.clf()
    plt.close()


def plot_scatter_pairwise(cor, model_name, display_name, study_label, color):
    """Scatter plot comparing per-perturbation correlations across training methods (reference = first method vs each method)."""
    methods = cor["training_method"].unique().tolist()
    if len(methods) < 2:
        return

    pivot = (
        cor.pivot_table(index="Gene", columns="training_method", values="Correlation", aggfunc="first")
        .reindex(columns=methods)
    )

    ref = methods[0]
    comparisons = methods[1:]
    n = len(comparisons)

    fig, axes = plt.subplots(1, n, figsize=(n * 5 / 2.54, 5 / 2.54), dpi=300)
    plt.rcParams.update({"font.size": 6, **_SOFT_RC})
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, comparisons):
        data = pivot[[ref, method]].dropna()
        vmin = min(data.min().min(), -0.1)
        vmax = max(data.max().max(), 1.0)
        lim = [vmin - 0.02, vmax + 0.02]
        ax.scatter(data[ref], data[method], s=2, alpha=0.5, color=color, linewidths=0)
        ax.plot(lim, lim, linestyle="--", color=LINE_COLOR, lw=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ref_label = ref.replace("\n", " ")
        method_label = method.replace("\n", " ")
        ax.set_xlabel(ref_label, fontsize=5)
        ax.set_ylabel(method_label, fontsize=5)
        r = data.corr().iloc[0, 1]
        ax.set_title(f"r={r:.3f}", fontsize=5)

    fig.suptitle(f"Perturbation-level correlation\n{study_label} / {display_name}", fontsize=6)
    plt.tight_layout()
    model_file = model_name.replace(" ", "_")
    study_file = study_label.replace(" ", "_").replace(".", "")
    plt.savefig(f"across_study/compare_training/Scatter_perts_{study_file}_{model_file}.svg")
    plt.clf()
    plt.close()


os.makedirs("across_study/compare_training", exist_ok=True)

for study_cfg in study_configs:
    study_name = study_cfg["name"]
    study_label = study_cfg["label"]
    print(f"=== {study_label} ===")

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        display_name = model_cfg.get("display_name", model_name)
        cor_acperts, cor_acgenes = load_correlations(study_name, model_cfg)

        if cor_acperts is None:
            print(f"  Skipping {model_name}: no data")
            continue

        plot_boxplot_by_exp(
            cor_acperts, model_name, display_name, study_label,
            palette=model_cfg["cmap"], wide=model_cfg["box_width"],
        )
        plot_barplot(
            cor_acgenes, model_name, display_name, study_label,
            color=model_cfg["bar_color"], wide=model_cfg["bar_width"],
        )
        plot_scatter_pairwise(
            cor_acperts, model_name, display_name, study_label,
            color=model_cfg["bar_color"],
        )
        print(f"  Plotted: {display_name}")
