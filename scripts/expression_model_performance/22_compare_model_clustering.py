#conda activate singlecell

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from dataset_model_config import (
    all_studies as studies,
    all_datanames as datanames,
    name_replace,
    pretrained_models,
    study_suffixes,
    model_fixed_suffix,
)


COMPARISON_LABELS = {
    "real_all - pred_all": "All set (Real vs Predicted)",
    "real_test - pred_test": "Test set (Real vs Predicted)",
    "real_all - real_test": "Real (All set vs Test set)",
    "pred_all - pred_test": "Predicted (All set vs Test set)",
}

COMPARISON_SLUGS = {
    "real_all - pred_all": "AllSet_RealVsPred",
    "real_test - pred_test": "TestSet_RealVsPred",
    "real_all - real_test": "Real_AllVsTestSet",
    "pred_all - pred_test": "Pred_AllVsTestSet",
}

EXCLUDE_STUDY_SUBSTRINGS = ("Wu2024",)

ALPHAGENOME_NAME = name_replace["alphagenome"]
ALPHAGENOME_COLOR = "#BC5765"
EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"

PAPER_FONT_SIZE = 7
HEATMAP_FONT_SIZE = 13  


def _apply_paper_rcparams(font_size=PAPER_FONT_SIZE):
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["legend.title_fontsize"] = font_size
    plt.rcParams["axes.edgecolor"] = EDGE_COLOR
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["xtick.color"] = EDGE_COLOR
    plt.rcParams["ytick.color"] = EDGE_COLOR
    plt.rcParams["xtick.labelcolor"] = "black"
    plt.rcParams["ytick.labelcolor"] = "black"
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6


def heatmap(clust_stats, output=""):
    out_dir = "across_study/compare_embedding"
    os.makedirs(out_dir, exist_ok=True)
    for comparison in set(clust_stats["Comparison"]):
        comp_label = COMPARISON_LABELS.get(comparison, comparison)
        comp_slug = COMPARISON_SLUGS.get(comparison, comparison.replace(" ", ""))
        for metrics, cmap, vmax in zip(
            ["NMI", "ARI", "FM"],
            ["viridis", "magma", "copper"],
            [0.6, 0.4, 0.4],
        ):
            df = clust_stats.query(
                'metrics == @metrics & Comparison == @comparison'
            ).pivot_table(
                index="pretrained_model", columns="Study", values="value", sort=False
            ).round(2)
            df = df.dropna(axis=1, how="any")
            n_studies = df.shape[1]
            n_models = df.shape[0]
            cell_cm = 1.4  
            width_cm = cell_cm * n_studies + 9.5  
            height_cm = cell_cm * n_models + 9.5  
            _apply_paper_rcparams(HEATMAP_FONT_SIZE)
            plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), dpi=300)
            ax = sns.heatmap(
                df, annot=True, linewidths=.5,
                vmax=vmax, vmin=0, cmap=cmap,
                square=True,
                annot_kws={"size": HEATMAP_FONT_SIZE},
                cbar_kws={"shrink": 0.7},
            )
            ax.set_ylabel("Model")
            ax.set_xlabel("Study")
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            for label in ax.get_yticklabels():
                label.set_rotation(0)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=HEATMAP_FONT_SIZE)
            ax.set_title(f"{comp_label} — {metrics}")
            plt.tight_layout()
            fname = f"{out_dir}/clustering_stats_{comp_slug}_{metrics}_{output}.svg"
            plt.savefig(fname)
            plt.clf()
            plt.close()


def barplot_alphagenome(clust_stats, output=""):
    out_dir = "across_study/compare_embedding"
    os.makedirs(out_dir, exist_ok=True)

    df_ag = clust_stats[
        (clust_stats["pretrained_model"] == ALPHAGENOME_NAME)
        & (clust_stats["metrics"].isin(["ARI", "NMI"]))
    ].copy()
    if df_ag.empty:
        return

    study_order = list(dict.fromkeys(df_ag["Study"].tolist()))
    metric_order = ["ARI", "NMI"]
    palette = {"ARI": ALPHAGENOME_COLOR, "NMI": "#E5B5BB"}

    for comparison in sorted(set(df_ag["Comparison"])):
        comp_label = COMPARISON_LABELS.get(comparison, comparison)
        comp_slug = COMPARISON_SLUGS.get(comparison, comparison.replace(" ", ""))
        sub = df_ag[df_ag["Comparison"] == comparison]
        if sub.empty:
            continue

        _apply_paper_rcparams()
        fig, ax = plt.subplots(figsize=(13 / 2.54, 6 / 2.54), dpi=300)
        sns.barplot(
            data=sub,
            x="Study", y="value",
            hue="metrics",
            order=study_order,
            hue_order=metric_order,
            palette=palette,
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            ax=ax,
        )
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Study")
        ax.set_ylabel("Score")
        ax.set_title(f"{ALPHAGENOME_NAME} clustering agreement — {comp_label}")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax.legend(
            title="Metric",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )
        sns.despine(ax=ax)
        plt.tight_layout()
        fname = f"{out_dir}/barplot_alphagenome_ARI_NMI_{comp_slug}_{output}.svg"
        plt.savefig(fname)
        plt.clf()
        plt.close()


def plot_heatmap(suffixes, models, output):
    clust_stats = pd.DataFrame()
    for study_suffix, model in zip(suffixes, models):
        for study_name, dataname in zip(studies, datanames):
            if any(s in study_name for s in EXCLUDE_STUDY_SUBSTRINGS):
                continue
            study = f'{study_name}__{study_suffix}'
            path = f"figures/{study}/embedding/clustering_metrics.txt"
            if not os.path.exists(path):
                continue
            clust_stat = pd.read_csv(path, sep="\t")
            clust_stat["pretrained_model"] = model
            clust_stats = pd.concat([clust_stats, clust_stat])
    clust_stats['pretrained_model'] = clust_stats['pretrained_model'].replace(name_replace)
    heatmap(clust_stats, output)
    barplot_alphagenome(clust_stats, output)


for study_suffix in study_suffixes:
    suffixes = []
    models = []
    for pretrained_model in pretrained_models:
        suffix = model_fixed_suffix.get(pretrained_model, study_suffix)
        suffixes.append(f"{pretrained_model}_{suffix}")
        models.append(pretrained_model)
    plot_heatmap(suffixes, models, "")
