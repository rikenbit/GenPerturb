import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
from dataset_model_config import (
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
    perturbation_list = {
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
    programs = pd.DataFrame()
    for i, j in perturbation_list.items():
        program = pd.DataFrame({"Perturbation": j})
        program["Program"] = i
        programs = pd.concat([programs, program])

    adata = adata.copy()
    adata.obs = pd.merge(adata.obs.reset_index(), programs, on="Perturbation", how="left").set_index('index')
    adata.obs["Program"] = adata.obs["Program"].fillna("others")

    program_order = list(perturbation_list.keys()) + ["others"]
    adata.obs["Program"] = pd.Categorical(
        adata.obs["Program"], categories=program_order, ordered=False
    )

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
    program_groups = list(perturbation_list.keys())
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



if __name__ == "__main__":
    study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"

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

            plot_program(ad_r_all, "real", study, suffix="_all", model_label=model_label)
            plot_program(ad_p_all, "pred", study, suffix="_all", model_label=model_label)
            plot_program(ad_r, "real", study, model_label=model_label)
            plot_program(ad_p, "pred", study, model_label=model_label)

            plot_signature(ad_r_all, "immune_differentiation", "real", study, model_label=model_label)
            plot_signature(ad_p_all, "immune_differentiation", "pred", study, model_label=model_label)

            combined_scores = calculate_and_save_signature_scores(ad_r_all, ad_p_all, study)
            plot_signature_scatter(combined_scores, study, model_label=model_label)

            print(f"[done] gene-signature: {study}")
