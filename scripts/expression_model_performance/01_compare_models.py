# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from dataset_model_config import (
    studies,
    xlabels,
    pretrained_models as pretrained_models_raw,
    cmaps as cmaps_raw,
    model_colors,
    name_replace,
    model_fixed_suffix,
)

# --- Load data for all models and studies ---
cor_acpertss_all = []
cor_acgeness_all = []

for pretrained_model in pretrained_models_raw:
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
        cor_acpertss_all.append(cor_acperts)
        cor_acgeness_all.append(cor_acgenes)

cor_acpertss_all = pd.concat(cor_acpertss_all, ignore_index=True)
cor_acgeness_all = pd.concat(cor_acgeness_all, ignore_index=True)

# Original short names (e.g. "AlphaGenome", "Borzoi", ...) — used for the legacy figures.
_short_name_replace = {
    "enformer": "Enformer",
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "alphagenome_fold_0": "AlphaGenome fold0",
    "alphagenome_fold_1": "AlphaGenome fold1",
    "alphagenome_fold_2": "AlphaGenome fold2",
    "alphagenome_fold_3": "AlphaGenome fold3",
    "enformerborzoi524k": "Enformer Borzoi 524k",
    "baseline_control": "Baseline Control",
    "baseline_peturbmean": "Baseline Perturbation Mean",
    "simplecnn": "Simple CNN",
}

# --- Plot per dataset ---
outdir = "across_study/compare_models/per_dataset"
os.makedirs(outdir, exist_ok=True)

EDGE_COLOR = "#B0B0B0"     # soft gray for spines and tick marks
BOX_LINE_COLOR = "#666666"  # darker gray for box/whisker/percentile lines
FONT_SIZE = 11
plt.rcParams.update({
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
})

def _plot_per_dataset(name_map, file_suffix, figsize):
    """Render the per-dataset figure with a given model-name mapping.

    name_map: dict mapping raw model id -> display name (e.g. "alphagenome" -> "AlphaGenome"
              for the legacy figure, or -> "GenPerturb (AlphaGenome transfer)" for the new one).
    file_suffix: appended to the saved filename (e.g. "" for legacy, "_genperturb" for new).
    figsize: matplotlib figsize tuple in inches.
    """
    cor_perts = cor_acpertss_all.copy()
    cor_genes = cor_acgeness_all.copy()
    cor_perts["model"] = cor_perts["model"].replace(name_map)
    cor_genes["model"] = cor_genes["model"].replace(name_map)
    display_models = [name_map.get(m, m) for m in pretrained_models_raw]
    cmap_dict = dict(zip(display_models, cmaps_raw))

    for xlabel in xlabels:
        df_genes = cor_genes.query(
            'training == "test" and study == @xlabel and model in @display_models'
        )
        df_perts = cor_perts.query(
            'training == "test" and study == @xlabel and model in @display_models'
        )

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300,
                                 gridspec_kw={"width_ratios": [0.8, 1.7]})

        # Left: barplot (correlation across genes)
        ax = axes[0]
        sns.barplot(
            data=df_genes,
            x="model",
            y="Correlation",
            order=display_models,
            palette=model_colors,
            saturation=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Model")
        ax.set_title("Correlation across genes")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")

        # Right: boxplot (correlation across perturbations, colored by expression level)
        ax = axes[1]
        hue_order = ["Very High", "High", "Medium", "Low", "Very Low"]
        n_hues = len(hue_order)
        box_width = 0.14
        for i, model in enumerate(display_models):
            model_data = df_perts.query('model == @model')
            colors = sns.color_palette(cmap_dict[model], n_hues, desat=0.75)
            for j, hue_val in enumerate(hue_order):
                subset = model_data.query('Mean == @hue_val')["Correlation"].dropna().values
                if len(subset) == 0:
                    continue
                pos = i + (j - n_hues / 2 + 0.5) * box_width
                bp = ax.boxplot([subset], positions=[pos], widths=box_width * 0.9,
                                patch_artist=True, showfliers=False, manage_ticks=False)
                bp['boxes'][0].set_facecolor(colors[j])
                bp['boxes'][0].set_edgecolor(BOX_LINE_COLOR)
                bp['boxes'][0].set_linewidth(0.7)
                for element in ['whiskers', 'caps', 'medians']:
                    for line in bp[element]:
                        line.set_linewidth(0.7)
                        line.set_color(BOX_LINE_COLOR)
        ax.set_xticks(range(len(display_models)))
        ax.set_xticklabels(display_models)
        margin = (n_hues / 2 + 0.5) * box_width
        ax.set_xlim(-margin, len(display_models) - 1 + margin)
        legend_patches = [Patch(facecolor='grey', alpha=1 - i * 0.18, edgecolor=BOX_LINE_COLOR, linewidth=0.5, label=l) for i, l in enumerate(hue_order)]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left',
                  borderaxespad=0., title="Expression")
        ymin = df_perts["Correlation"].min() if len(df_perts) > 0 else 0
        ax.set_ylim(min(ymin, 0), 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
        ax.set_xlabel("Model")
        ax.set_title("Correlation across perturbations")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")

        plt.tight_layout()
        plt.subplots_adjust(top=0.81)
        fig.suptitle(xlabel, y=0.96)

        safe_name = xlabel.replace(" ", "_").replace(".", "")
        plt.savefig(f"{outdir}/{safe_name}{file_suffix}.svg")
        plt.clf()
        plt.close()


# Legacy figure (short names like "AlphaGenome"): smaller / original layout.
_plot_per_dataset(
    name_map=_short_name_replace,
    file_suffix="",
    figsize=(17 / 2.54, 7 / 2.54),
)
# New figure with GenPerturb-prefixed names: needs more room for the longer labels.
_plot_per_dataset(
    name_map=name_replace,
    file_suffix="_genperturb",
    figsize=(22 / 2.54, 11 / 2.54),
)
