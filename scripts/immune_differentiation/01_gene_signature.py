import os
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
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


def _apply_soft_axes(ax, square=True):
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COLOR)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=EDGE_COLOR, width=0.6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("black")
    if square:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_box_aspect(1)


PROGRAM_COLORS = {
    "Erythroid": "#C0392B",
    "Granulocyte": "#3498DB",
    "Megakaryocyte": "#7E57C2",
    "Pioneer_factor": "#F39C12",
    "Pro_growth": "#43A047",
    "G1_cycle": "#795548",
    "others": "#BDBDBD",
}


# Curated perturbation -> transcriptional program assignment (Norman et al.).
# Used both for the UMAP program overlay and as the reference labelling for the
# program-based clustering agreement (ARI / NMI / FM) computed below.
PERTURBATION_PROGRAMS = {
    "Erythroid": ['CBL_CNN1', 'CBL_PTPN12', 'CBL_PTPN9', 'CBL_UBASH3B', 'SAMD1_PTPN12',
                  'SAMD1_UBASH3B', 'UBASH3B_CNN1', 'UBASH3B_PTPN12', 'UBASH3B_PTPN9', 'UBASH3B_UBASH3A',
                  'UBASH3B_ZBTB25', 'BPGM_SAMD1', 'PTPN1', 'PTPN12_PTPN9', 'PTPN12_UBASH3A', 'PTPN12_ZBTB25'],
    "Granulocyte": ['SPI1', 'CEBPA', 'CEBPB', 'CEBPE_CEBPA', 'CEBPE_RUNX1T1', 'CEBPE_SPI1',
                    'CEBPE', 'ETS2_CEBPE', 'KLF1_CEBPA', 'FOSB_CEBPE'],
    "Megakaryocyte": ['MAPK1_TGFBR2', 'MAPK1', 'ETS2_MAPK1', 'ETS2', 'CEBPB_MAPK1'],
    "Pioneer_factor": ['FOXA1_FOXF1', 'FOXA1_FOXL2', 'FOXA1_HOXB9', 'FOXA3_FOXA1',
                       'FOXA3_FOXF1', 'FOXA3_FOXL2', 'FOXA3_HOXB9', 'FOXA3', 'FOXF1_FOXL2', 'FOXF1_HOXB9',
                       'FOXL2_MEIS1', 'HOXA13', 'HOXC13', 'POU3F2_FOXL2', 'TP73', 'MIDN', 'LYL1_IER5L',
                       'DUSP9_SNAI1', 'ZBTB10_SNAI1'],
    "Pro_growth": ['CEBPE_KLF1', 'KLF1', 'KLF1_BAK1', 'KLF1_TGFBR2', 'ELMSAN1', 'MAP2K3_SLC38A2',
                   'MAP2K3_ELMSAN1', 'MAP2K3', 'MAP2K3_MAP2K6', 'MAP2K6_ELMSAN1', 'MAP2K6', 'KLF1_MAP2K6'],
    "G1_cycle": ['CDKN1A', 'CDKN1B_CDKN1A', 'CDKN1B', 'CDKN1C_CDKN1A', 'CDKN1C'],
}

PROGRAM_ORDER = list(PERTURBATION_PROGRAMS.keys()) + ["others"]


def annotate_program(adata):
    """Return a copy of adata with obs['Program'] from PERTURBATION_PROGRAMS."""
    programs = pd.DataFrame()
    for i, j in PERTURBATION_PROGRAMS.items():
        program = pd.DataFrame({"Perturbation": j})
        program["Program"] = i
        programs = pd.concat([programs, program])

    adata = adata.copy()
    adata.obs = pd.merge(adata.obs.reset_index(), programs, on="Perturbation", how="left").set_index('index')
    adata.obs["Program"] = adata.obs["Program"].fillna("others")
    adata.obs["Program"] = pd.Categorical(
        adata.obs["Program"], categories=PROGRAM_ORDER, ordered=False
    )
    return adata


def load_adata(study):
    if os.path.exists(f'adata/{study}/adata_real_all.h5ad'):
        adata_real_all = sc.read_h5ad(f'adata/{study}/adata_real_all.h5ad')
        adata_pred_all = sc.read_h5ad(f'adata/{study}/adata_pred_all.h5ad')
        adata_real = sc.read_h5ad(f'adata/{study}/adata_real.h5ad')
        adata_pred = sc.read_h5ad(f'adata/{study}/adata_pred.h5ad')
        return adata_real_all, adata_pred_all, adata_real, adata_pred
    else:
        raise FileNotFoundError(f"adata files not found for {study}")


## gene signature plot for Norman
def plot_program(adata, value_type, study, suffix="", model_label=""):
    adata = annotate_program(adata)
    program_order = PROGRAM_ORDER

    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)

    sc.set_figure_params(fontsize=6, dpi=300, dpi_save=300)
    plt.rcParams.update({
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    figsize = (7.6 / 2.54, 4.0 / 2.54)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    program_groups = list(PERTURBATION_PROGRAMS.keys())
    palette = [PROGRAM_COLORS.get(g, "#888888") for g in program_order]
    sc.pl.umap(
        adata, color="Program", s=30,
        groups=program_groups,
        palette=palette,
        ax=ax, show=False,
    )
    _apply_soft_axes(ax, square=True)
    value_type_label = "Observed" if value_type == "real" else "Predicted"
    title_lines = [f"Norman et al. — Gene signature ({value_type_label})"]
    if model_label:
        title_lines.append(model_label)
    ax.set_title("\n".join(title_lines), fontsize=6)
    ax.legend(scatterpoints=1, markerscale=0.2, loc='upper left',
              bbox_to_anchor=(1.05, 1), fontsize=6, title="Program",
              title_fontsize=6)
    plt.tight_layout()
    outpath = f'{outdir}/{value_type}_program{suffix}.svg'
    plt.savefig(outpath)
    print(f"[debug] Saved: {outpath}")
    plt.clf()
    plt.close()


def plot_signature(adata, signature, value_type, study, model_label=""):
    signature_list = {
        "Erythroid": ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "GYPA", "ERMAP"],
        "Granulocyte": ["ITGAM", "CSF3R", "LST1", "CD33"],
        "Megakaryocyte": ["ITGA2B"]
    }

    adata = adata.copy()
    for i, j in signature_list.items():
        genes_exist = [g for g in j if g in adata.var_names]
        if len(genes_exist) > 0:
            sc.tl.score_genes(adata, genes_exist, score_name=i)
        else:
            adata.obs[i] = 0

    signature_num = len(signature_list.keys())
    sc.set_figure_params(fontsize=6, dpi=300, dpi_save=300)
    plt.rcParams.update({
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    fig, axes = plt.subplots(1, signature_num, figsize=(4.6 / 2.54 * signature_num, 4.5 / 2.54), dpi=300)

    value_type_label = "Observed" if value_type == "real" else "Predicted"
    for i, score in enumerate(signature_list.keys()):
        maxabs = max(abs(adata.obs[score].max()), abs(adata.obs[score].min()), 0.001)
        sc.pl.umap(adata, color=score, cmap="RdBu_r", s=30, vmin=-maxabs, vmax=maxabs, vcenter=0,
                   ax=axes[i], legend_fontsize="xx-small", show=False)
        _apply_soft_axes(axes[i], square=True)
        axes[i].set_title(f"{score}\n({value_type_label})", fontsize=6)

    suptitle = f"Norman et al. — {value_type_label}"
    if model_label:
        suptitle = f"{suptitle} | {model_label}"
    fig.suptitle(suptitle, fontsize=7)
    plt.tight_layout()
    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/{value_type}_{signature}.svg'
    plt.savefig(outpath)
    print(f"[debug] Saved: {outpath}")
    plt.clf()
    plt.close()


def calculate_and_save_signature_scores(adata_real, adata_pred, study):
    signature_list = {
        "Erythroid": ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "GYPA", "ERMAP"],
        "Granulocyte": ["ITGAM", "CSF3R", "LST1", "CD33"],
        "Megakaryocyte": ["ITGA2B"]
    }

    adata_real = adata_real.copy()
    adata_pred = adata_pred.copy()

    for sig_name, genes in signature_list.items():
        genes_exist = [g for g in genes if g in adata_real.var_names]
        if len(genes_exist) > 0:
            sc.tl.score_genes(adata_real, genes_exist, score_name=sig_name)
        else:
            adata_real.obs[sig_name] = 0

    for sig_name, genes in signature_list.items():
        genes_exist = [g for g in genes if g in adata_pred.var_names]
        if len(genes_exist) > 0:
            sc.tl.score_genes(adata_pred, genes_exist, score_name=sig_name)
        else:
            adata_pred.obs[sig_name] = 0

    real_scores = adata_real.obs[list(signature_list.keys())].copy()
    real_scores['Perturbation'] = adata_real.obs['Perturbation'].values
    real_scores['value_type'] = 'real'

    pred_scores = adata_pred.obs[list(signature_list.keys())].copy()
    pred_scores['Perturbation'] = adata_pred.obs['Perturbation'].values
    pred_scores['value_type'] = 'pred'

    combined_scores = pd.concat([real_scores, pred_scores])

    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/signature_scores.txt'
    combined_scores.to_csv(outpath, sep='\t', index=False)
    print(f"[debug] Saved: {outpath}")

    return combined_scores


def plot_signature_scatter(combined_scores, study, model_label=""):
    signature_list = ["Erythroid", "Granulocyte", "Megakaryocyte"]

    real_df = combined_scores[combined_scores['value_type'] == 'real'].set_index('Perturbation')
    pred_df = combined_scores[combined_scores['value_type'] == 'pred'].set_index('Perturbation')

    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.grid": False,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })

    for sig_name in signature_list:
        fig, ax = plt.subplots(figsize=(5 / 2.54, 5 / 2.54), dpi=300)

        common_perts = real_df.index.intersection(pred_df.index)
        if len(common_perts) == 0:
            print(f"[skip] No common perturbations for {sig_name}")
            plt.close(fig)
            continue

        x_vals = real_df.loc[common_perts, sig_name].values
        y_vals = pred_df.loc[common_perts, sig_name].values

        ax.scatter(x_vals, y_vals, s=10, alpha=0.6,
                   edgecolors=LINE_COLOR, linewidth=0.3,
                   color=PROGRAM_COLORS.get(sig_name, "#1f77b4"))

        if len(x_vals) > 1:
            r, p = pearsonr(x_vals, y_vals)
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=6, verticalalignment='top')

        ax.set_xlabel('Observed', fontsize=6)
        ax.set_ylabel('Predicted', fontsize=6)
        title = sig_name if not model_label else f"{sig_name}\n{model_label}"
        ax.set_title(title, fontsize=6)
        ax.tick_params(labelsize=5)
        _apply_soft_axes(ax, square=True)

        plt.tight_layout()
        outpath = f'{outdir}/signature_scatter_{sig_name}.svg'
        plt.savefig(outpath)
        print(f"[debug] Saved: {outpath}")
        plt.clf()
        plt.close()

    print(f"[done] gene signature scatter plot: {study}")


def _leiden_key(adata, value_type):
    """Return the leiden column this adata carries ('leiden_real'/'leiden_pred')."""
    for key in (f"leiden_{value_type}", "leiden"):
        if key in adata.obs.columns:
            return key
    return None


def calculate_program_clustering_metrics(adata_real, adata_pred, study, dataname="", subset="all"):
    """Score leiden clusterings against the curated program annotation.

    Unlike 21_embedding_signature.py, which compares the observed and predicted
    leiden partitions to each other, this uses the perturbation-list based
    program labels (Erythroid, G1_cycle, ...) as the reference partition and
    asks how well the observed and the predicted clusterings each recover it.
    Reported for two labellings: all perturbations (unannotated ones pooled as
    'others') and annotated perturbations only.
    """
    key_real = _leiden_key(adata_real, "real")
    key_pred = _leiden_key(adata_pred, "pred")
    if key_real is None or key_pred is None:
        print(f"[skip] leiden columns missing for {study} ({subset} set)")
        return pd.DataFrame()

    ad_real = annotate_program(adata_real)
    ad_pred = annotate_program(adata_pred)

    common = ad_real.obs_names.intersection(ad_pred.obs_names)
    if len(common) == 0:
        print(f"[skip] no shared perturbations for {study} ({subset} set)")
        return pd.DataFrame()

    obs_real = ad_real.obs.loc[common]
    obs_pred = ad_pred.obs.loc[common]

    program = obs_real["Program"].astype(str)
    clusters = {
        "leiden_real": obs_real[key_real].astype(str),
        "leiden_pred": obs_pred[key_pred].astype(str),
    }

    label_sets = {
        "with_others": pd.Series(True, index=common),
        "annotated_only": (program != "others").values,
    }

    records = []
    for label_set, mask in label_sets.items():
        ref = program[mask]
        if ref.nunique() < 2:
            print(f"[skip] <2 programs for {study} ({subset}, {label_set})")
            continue
        for comparison, clust in clusters.items():
            obs_clust = clust[mask]
            records.append({
                "Study": dataname or study,
                "Set": subset,
                "Labels": label_set,
                "Comparison": f"program - {comparison}",
                "n_perturbations": int(len(ref)),
                "n_programs": int(ref.nunique()),
                "n_clusters": int(obs_clust.nunique()),
                "ARI": adjusted_rand_score(ref, obs_clust),
                "NMI": normalized_mutual_info_score(ref, obs_clust),
                "FM": fowlkes_mallows_score(ref, obs_clust),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    id_cols = ["Study", "Set", "Labels", "Comparison",
               "n_perturbations", "n_programs", "n_clusters"]
    df = (
        df.set_index(id_cols)
        .stack()
        .reset_index()
        .rename(columns={f"level_{len(id_cols)}": "metrics", 0: "value"})
    )
    return df


def plot_program_clustering_barplot(metrics_df, study, model_label=""):
    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)

    palette = {"leiden_real": "#4C72B0", "leiden_pred": "#BC5765"}
    hue_order = ["program - leiden_real", "program - leiden_pred"]

    for label_set in metrics_df["Labels"].unique():
        sub = metrics_df[
            (metrics_df["Labels"] == label_set)
            & (metrics_df["metrics"].isin(["ARI", "NMI"]))
        ]
        if sub.empty:
            continue
        sub = sub.assign(group=sub["Set"] + " / " + sub["metrics"])

        plt.rcParams.update({
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "axes.grid": False,
            "axes.edgecolor": EDGE_COLOR,
            "axes.linewidth": 0.6,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
        })
        fig, ax = plt.subplots(figsize=(8 / 2.54, 5 / 2.54), dpi=300)
        sns.barplot(
            data=sub, x="group", y="value", hue="Comparison",
            hue_order=[h for h in hue_order if h in set(sub["Comparison"])],
            palette={f"program - {k}": v for k, v in palette.items()},
            edgecolor=EDGE_COLOR, linewidth=0.4, ax=ax,
        )
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Set / metric")
        ax.set_ylabel("Score vs program annotation")
        title = f"Norman et al. — program clustering agreement ({label_set})"
        if model_label:
            title = f"{title}\n{model_label}"
        ax.set_title(title)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0, frameon=False, fontsize=6)
        sns.despine(ax=ax)
        plt.tight_layout()
        outpath = f'{outdir}/program_clustering_metrics_{label_set}.svg'
        plt.savefig(outpath)
        print(f"[debug] Saved: {outpath}")
        plt.clf()
        plt.close()


def run_program_clustering_metrics(ad_r_all, ad_p_all, ad_r, ad_p, study,
                                   dataname="", model_label=""):
    metrics_df = pd.concat([
        calculate_program_clustering_metrics(ad_r_all, ad_p_all, study,
                                             dataname=dataname, subset="all"),
        calculate_program_clustering_metrics(ad_r, ad_p, study,
                                             dataname=dataname, subset="test"),
    ])
    if metrics_df.empty:
        print(f"[skip] no program clustering metrics for {study}")
        return metrics_df

    outdir = f'figures/{study}/gene_signature'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/program_clustering_metrics.txt'
    metrics_df.to_csv(outpath, sep="\t", index=False)
    print(f"[debug] Saved: {outpath}")

    plot_program_clustering_barplot(metrics_df, study, model_label=model_label)
    return metrics_df


if __name__ == "__main__":
    study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
    dataname = all_datanames[all_studies.index(study_name)] if study_name in all_studies else study_name

    # ONLY_PROGRAM_CLUSTERING=1 recomputes just the program-based ARI/NMI/FM and
    # its barplots, leaving the existing UMAP / signature outputs untouched.
    only_program_clustering = bool(os.environ.get("ONLY_PROGRAM_CLUSTERING"))

    program_clustering_summary = pd.DataFrame()

    for model in pretrained_models:
        suffixes = [model_fixed_suffix[model]] if model in model_fixed_suffix else study_suffixes
        for suf in suffixes:
            study = f"{study_name}__{model}_{suf}"
            model_label = name_replace.get(model, model)

            try:
                ad_r_all, ad_p_all, ad_r, ad_p = load_adata(study)
            except Exception as e:
                print(f"[skip] {study} – {e}")
                continue

            gene_dir = f"figures/{study}/gene_signature"
            os.makedirs(gene_dir, exist_ok=True)

            if not only_program_clustering:
                plot_program(ad_r_all, "real", study, suffix="_all", model_label=model_label)
                plot_program(ad_p_all, "pred", study, suffix="_all", model_label=model_label)
                plot_program(ad_r, "real", study, model_label=model_label)
                plot_program(ad_p, "pred", study, model_label=model_label)

                plot_signature(ad_r_all, "immune_differentiation", "real", study, model_label=model_label)
                plot_signature(ad_p_all, "immune_differentiation", "pred", study, model_label=model_label)

                combined_scores = calculate_and_save_signature_scores(ad_r_all, ad_p_all, study)
                plot_signature_scatter(combined_scores, study, model_label=model_label)

            metrics_df = run_program_clustering_metrics(
                ad_r_all, ad_p_all, ad_r, ad_p, study,
                dataname=dataname, model_label=model_label,
            )
            if not metrics_df.empty:
                metrics_df = metrics_df.copy()
                metrics_df.insert(1, "pretrained_model", model_label)
                metrics_df.insert(2, "study_dir", study)
                program_clustering_summary = pd.concat(
                    [program_clustering_summary, metrics_df]
                )

            print(f"[done] gene-signature: {study}")

    if not program_clustering_summary.empty:
        summary_dir = "across_study/gene_signature"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = f"{summary_dir}/program_clustering_metrics_all_models.txt"
        program_clustering_summary.to_csv(summary_path, sep="\t", index=False)
        print(f"[debug] Saved: {summary_path}")
