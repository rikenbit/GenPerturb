#conda activate singlecell

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from adjustText import adjust_text
import PIL.Image as Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import os

from dataset_model_config import (
    all_studies,
    all_datanames,
    pretrained_models,
    study_suffixes,
    model_fixed_suffix,
    name_replace,
)

EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"


def _apply_soft_axes(ax):
    """Force axis spines / ticks to soft gray.

    scanpy.pl.umap sets spine color to black explicitly, so we override
    after the call. Also enforce a square axes box for UMAP figures.
    """
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=EDGE_COLOR, width=0.6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("black")

_target_env = os.environ.get("TARGET_STUDIES")
if _target_env:
    _names = [s.strip() for s in _target_env.split(",") if s.strip()]
    studies = []
    datanames = []
    for _n in _names:
        if _n not in all_studies:
            print(f"[warn] unknown study: {_n}; skipping")
            continue
        _i = all_studies.index(_n)
        studies.append(all_studies[_i])
        datanames.append(all_datanames[_i])
else:
    studies = list(all_studies)
    datanames = list(all_datanames)



def pre_processing(adata, value_type="pred"):
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.leiden(adata, resolution=1.5, key_added=f"leiden_{value_type}")
    sc.tl.umap(adata)
    return adata

def load_adata(study):
    if os.path.exists(f'adata/{study}/adata_real_all.h5ad'):
        adata_real_all = sc.read_h5ad(f'adata/{study}/adata_real_all.h5ad')
        adata_pred_all = sc.read_h5ad(f'adata/{study}/adata_pred_all.h5ad')
        adata_real = sc.read_h5ad(f'adata/{study}/adata_real.h5ad')
        adata_pred = sc.read_h5ad(f'adata/{study}/adata_pred.h5ad')
    return adata_real_all, adata_pred_all, adata_real, adata_pred

def preprocess_adata(study_name, study):
    df = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
    pred = np.load(f"prediction/{study}/prediction.npy")
    df2 = pd.DataFrame(pred)
    df2.columns = df.columns[1:]
    df2.index = df.index
    adata_real_all = AnnData(df.drop("training", axis=1).T)
    adata_pred_all = AnnData(df2.T)
    df_test = df.query('training == "test"')
    adata_real = AnnData(df_test.drop("training", axis=1).T)
    adata_pred = AnnData(df2.loc[df_test.index,:].T)
    for adata in [adata_real_all, adata_pred_all, adata_real, adata_pred]:
        adata.var_names_make_unique()
        adata.obs[["study", "Perturbation"]] = [i.split(".", 1) for i in adata.obs.index]
        adata.obs["Perturbation"] = adata.obs["Perturbation"].astype("category")
        adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X
    pre_processing(adata_real_all, value_type="real")
    pre_processing(adata_pred_all, value_type="pred")
    pre_processing(adata_real, value_type="real")
    pre_processing(adata_pred, value_type="pred")
    sc.tl.rank_genes_groups(adata_real_all, groupby=f"leiden_real")
    sc.tl.rank_genes_groups(adata_pred_all, groupby=f"leiden_pred")
    sc.tl.rank_genes_groups(adata_real, groupby=f"leiden_real")
    sc.tl.rank_genes_groups(adata_pred, groupby=f"leiden_pred")
    sc.tl.dendrogram(adata_real_all, groupby=f"leiden_real")
    sc.tl.dendrogram(adata_pred_all, groupby=f"leiden_pred")
    sc.tl.dendrogram(adata_real, groupby=f"leiden_real")
    sc.tl.dendrogram(adata_pred, groupby=f"leiden_pred")
    return adata_real_all, adata_pred_all, adata_real, adata_pred

def create_umap_plot(adata, value_type="", dataname="", leiden_key="leiden", suffix="",
                      model_label="", on_text=False):
    cluster_num = len(set(adata.obs[leiden_key]))
    ncol = (cluster_num - 1) // 9 + 1
    plt.rcParams.update({
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    # Square plotting axes (4.2cm x 4.2cm); add room for legend on the right.
    figsize = ((4.2 + 1.3 * ncol)/2.54, 4.2/2.54)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    leiden_origin = "Observed" if leiden_key.endswith("_real") else "Predicted"
    title_lines = [dataname]
    if model_label:
        title_lines.append(model_label)
    title_lines.append(f'{leiden_key.replace("_", " ")} (umap {value_type})')
    sc.pl.umap(adata, color=leiden_key, size=20, ax=ax,
               title="\n".join(title_lines), legend_fontsize="xx-small", show=False)
    # Force a square box AND equal data scaling (scanpy returns a stretched
    # box on its own, and overrides spine colors back to black).
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    _apply_soft_axes(ax)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    ax.legend(scatterpoints=1, markerscale=0.2, fontsize=6, ncol=ncol,
              loc='upper left', bbox_to_anchor=(1.05, 1),
              title=f"{leiden_origin} leiden", title_fontsize=6)
    plt.tight_layout()
    plt.savefig(f'figures/{study}/embedding/{value_type}_{leiden_key}{suffix}.svg')
    plt.clf()
    plt.close()
    if on_text:
        plt.rcParams["font.size"] = 5
        plt.rcParams["axes.titlesize"] = 5
        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.umap(adata, color=leiden_key, size=10, ax=ax,
                   title="\n".join(title_lines), show=False)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_box_aspect(1)
        _apply_soft_axes(ax)
        ax.xaxis.label.set_size(5)
        ax.yaxis.label.set_size(5)
        for i in range(adata.n_obs):
            ax.text(adata.obsm["X_umap"][i][0], adata.obsm["X_umap"][i][1], adata.obs["Perturbation"][i], fontdict={"size": 6}, ha="center", va="center")
        adjust_text(ax.texts, arrowprops=dict(arrowstyle="-", color=LINE_COLOR))
        plt.tight_layout()
        plt.savefig(f'figures/{study}/embedding/{value_type}_{leiden_key}{suffix}_ondata.svg')
        plt.clf()
        plt.close()
    return adata

def transfer_real_info(adata_real, adata_pred):
    adata_pred.obs["leiden_real"] = adata_real.obs["leiden_real"]
    adata_pred.uns["dendrogram_leiden_real"] = adata_real.uns["dendrogram_leiden_real"]
    adata_pred.uns["leiden_real_colors"] = adata_real.uns["leiden_real_colors"]
    return adata_real, adata_pred

def save_adata(adata_real_all, adata_pred_all, adata_real, adata_pred):
    adata_real_all.write(f'adata/{study}/adata_real_all.h5ad')
    adata_pred_all.write(f'adata/{study}/adata_pred_all.h5ad')
    adata_real.write(f'adata/{study}/adata_real.h5ad')
    adata_pred.write(f'adata/{study}/adata_pred.h5ad')

def calculate_cluster_stats(adata_pred_all, adata_pred, dataname, study):
    clust_real_all = list(adata_pred_all.obs["leiden_real"])
    clust_pred_all = list(adata_pred_all.obs["leiden_pred"])
    clust_real = list(adata_pred.obs["leiden_real"])
    clust_pred = list(adata_pred.obs["leiden_pred"])
    clust_stats_all = {
        "Study": [dataname],
        "Comparison": ["real_all - pred_all"],
        "ARI": [adjusted_rand_score(clust_real_all, clust_pred_all)],
        "NMI": [normalized_mutual_info_score(clust_real_all, clust_pred_all)],
        "FM":  [fowlkes_mallows_score(clust_real_all, clust_pred_all)]
    }
    clust_stats_test = {
        "Study": [dataname],
        "Comparison": ["real_test - pred_test"],
        "ARI": [adjusted_rand_score(clust_real, clust_pred)],
        "NMI": [normalized_mutual_info_score(clust_real, clust_pred)],
        "FM":  [fowlkes_mallows_score(clust_real, clust_pred)]
    }
    clust_stats_real = {
        "Study": [dataname],
        "Comparison": ["real_all - real_test"],
        "ARI": [adjusted_rand_score(clust_real_all, clust_real)],
        "NMI": [normalized_mutual_info_score(clust_real_all, clust_real)],
        "FM":  [fowlkes_mallows_score(clust_real_all, clust_real)]
    }
    clust_stats_pred = {
        "Study": [dataname],
        "Comparison": ["pred_all - pred_test"],
        "ARI": [adjusted_rand_score(clust_pred_all, clust_pred)],
        "NMI": [normalized_mutual_info_score(clust_pred_all, clust_pred)],
        "FM":  [fowlkes_mallows_score(clust_pred_all, clust_pred)]
    }
    clust_stats_summary = pd.DataFrame()
    clust_stats_summary = pd.concat([clust_stats_summary, pd.DataFrame(clust_stats_all )])
    clust_stats_summary = pd.concat([clust_stats_summary, pd.DataFrame(clust_stats_test)])
    clust_stats_summary = pd.concat([clust_stats_summary, pd.DataFrame(clust_stats_real)])
    clust_stats_summary = pd.concat([clust_stats_summary, pd.DataFrame(clust_stats_pred)])
    clust_stats_summary = clust_stats_summary.set_index(["Study", "Comparison"]).stack().reset_index().rename(columns={"level_2":"metrics", 0:"value"})
    clust_stats_summary.to_csv(f"figures/{study}/embedding/clustering_metrics.txt", sep="\t", index=False)
    return clust_stats_summary

def plot_barplot(clust_stats_summary, study):
    plt.figure(figsize=(8/2.54, 5/2.54), dpi=300)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 5
    g = sns.barplot(
        data=clust_stats_summary,
        x="Comparison",
        y="value",
        hue="metrics",
        palette="Dark2",
        order=["real_all - pred_all", "real_test - pred_test", "real_all - real_test", "pred_all - pred_test"],
        hue_order=["ARI", "NMI", "FM"],
    )
    g.set_title(clust_stats_summary["Study"].iloc[0])
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{study}/embedding/clustering_metrics.svg")
    plt.clf()
    plt.close()

def plot_heatmap(adata, study, value_type, rank,  n_genes=10, layer="scaled", vmax=2, vmin=-2, suffix=""):
    plt.figure(figsize=(16/2.54, 12/2.54), dpi=300)
    plt.rcParams["font.size"] = 5
    sc.pl.rank_genes_groups_heatmap(adata,
        n_genes=n_genes, layer=layer, 
        vmax=vmax, vmin=vmin, 
        show_gene_labels=True, 
        use_raw=False, figsize=(16/2.54, 12/2.54), 
        cmap="viridis", key=f"rank_genes_groups_curated_{rank}"
    )
    plt.savefig(f"figures/{study}/embedding/{value_type}_markergenes_{rank}{suffix}.svg")
    plt.clf()
    plt.close()


for pretrained_model in pretrained_models:
    suffixes = [model_fixed_suffix[pretrained_model]] if pretrained_model in model_fixed_suffix else study_suffixes
    for study_suffix in suffixes:
        for study_name, dataname in zip(studies, datanames):
            study = f"{study_name}__{pretrained_model}_{study_suffix}"
            print(study)

            if not os.path.exists(f"figures/{study}"):
                continue
            # Skip if already done unless FORCE_REGENERATE is set; that lets
            # users re-render the embedding figures (e.g. after a style change)
            # without deleting old outputs by hand.
            if (
                os.path.exists(f"figures/{study}/embedding")
                and not os.environ.get("FORCE_REGENERATE")
            ):
                print(f"[skip] already finished: {study}")
                continue

            os.makedirs(f"adata/{study}", exist_ok=True)

            if os.path.exists(f"adata/{study}/adata_pred.h5ad"):
                adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)
            else:
                adata_real_all, adata_pred_all, adata_real, adata_pred = preprocess_adata(study_name, study)
                save_adata(adata_real_all, adata_pred_all, adata_real, adata_pred)

            os.makedirs(f"figures/{study}/embedding", exist_ok=True)

            model_label = name_replace.get(pretrained_model, pretrained_model)

            adata_real_all = create_umap_plot(adata_real_all, value_type="real", dataname=dataname, leiden_key="leiden_real", suffix="_all", model_label=model_label)
            adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_pred", suffix="_all", model_label=model_label)
            adata_real     = create_umap_plot(adata_real,     value_type="real", dataname=dataname, leiden_key="leiden_real", model_label=model_label)
            adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_pred", model_label=model_label)

            transfer_real_info(adata_real_all, adata_pred_all)
            transfer_real_info(adata_real,     adata_pred)

            adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_real", suffix="_all", model_label=model_label)
            adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_real", model_label=model_label)

            clust_stats_summary = calculate_cluster_stats(adata_pred_all, adata_pred, dataname, study)
            plot_barplot(clust_stats_summary, study)




