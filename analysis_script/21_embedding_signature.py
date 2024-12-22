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
import decoupler as dc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import average_precision_score
import os



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

def create_umap_plot(adata, value_type="", dataname="", leiden_key="leiden", suffix="", on_text=False):
    cluster_num = len(set(adata.obs[leiden_key]))
    ncol = (cluster_num - 1) // 9 + 1
    figsize = ((3.9 + 1.3 * ncol)/2.54, 4.2/2.54)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    plt.rcParams["font.size"] = 6
    plt.rcParams["axes.titlesize"] = 6
    sc.pl.umap(adata, color=leiden_key, size=20, ax=ax, title=f'{dataname}\n{leiden_key.replace("_", " ")} (umap {value_type})', legend_fontsize="xx-small")
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    ax.legend(scatterpoints=1, markerscale=0.2, fontsize=6, ncol=ncol, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(f'figures/{study}/embedding/{value_type}_{leiden_key}{suffix}.svg')
    plt.clf()
    plt.close()
    if on_text:
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["font.size"] = 5
        plt.rcParams["axes.titlesize"] = 5
        sc.pl.umap(adata, color=leiden_key, size=10, ax=ax, title=f'{dataname}\n{leiden_key.replace("_", " ")} (umap {value_type})')
        ax.xaxis.label.set_size(5)
        ax.yaxis.label.set_size(5)
        for i in range(adata.n_obs):
            ax.text(adata.obsm["X_umap"][i][0], adata.obsm["X_umap"][i][1], adata.obs["Perturbation"][i], fontdict={"size": 6}, ha="center", va="center")
        adjust_text(ax.texts, arrowprops=dict(arrowstyle="-", color="black"))
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

def curate_rank_genes(adata, value_type):
    adata_curated = adata[:,adata.var.index.isin(cor_genes)].copy()
    sc.tl.rank_genes_groups(adata_curated, groupby=f"leiden_{value_type}")
    adata.uns[f"rank_genes_groups_curated_{value_type}"] = adata_curated.uns["rank_genes_groups"]

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

## studies
studies = [
"NormanWeissman2019_filtered_mixscape_exnp_train",
"ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
"ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
"ReplogleWeissman2022_rpe1_mixscape_exnp_train",
"JialongJiang2024_Myeloid_train",
"JialongJiang2024_CD4T_train",
"JialongJiang2024_CD8T_train",
"JialongJiang2024_B_cell_train",
"JuliaJoung2023_TFAtlas_train",
"Srivatsan2019_A549_train",
"Srivatsan2019_K562_train",
"Srivatsan2019_MCF7_train",
]

datanames = [
"Norman et al.",
"Replogle et al. essential",
"Replogle et al. gwps",
"Replogle et al. rpe1",
"Jialong et al. myeloid",
"Jialong et al. CD4T",
"Jialong et al. CD8T",
"Jialong et al. Bcell",
"Julia et al.",
"Srivatsan et al. A549",
"Srivatsan et al. K562",
"Srivatsan et al. MCF7",
]

pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"hyena_dna_mean",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
"nucleotide_transformer_mean",
]

study_suffixes = [
"transfer_epoch100_batch256_adamw5e3",
"transfer_epoch100_batch256_adamw5e3_fromdna",
"finetuning_epoch20_batch8_adamw5e3_lora_r16a2",
"finetuning_epoch20_batch8_adamw5e3_lora_r64a2",
"finetuning_epoch20_batch8_adamw5e3_lora_r256a2",
"finetuning_epoch20_batch8_adamw5e3_lora_r512a2",
"finetuning_epoch20_batch8_adamw5e3_lora_r1024a2",
"finetuning_epoch40_batch2_adamw5e3_full",
"finetuning_epoch40_batch2_adamw5e3_full_plr1e10",
]


for pretrained_model in pretrained_models:
    for study_suffix in study_suffixes:
        for study_name, dataname in zip(studies, datanames):
            study = f'{study_name}__{pretrained_model}_{study_suffix}'
            print(study)
            if not os.path.exists(f'figures/{study}'):
                continue
            elif os.path.exists(f'adata/{study}/adata_pred.h5ad'):
                continue
            ## Load data
            os.makedirs(f'adata/{study}', exist_ok=True)
            adata_real_all, adata_pred_all, adata_real, adata_pred = preprocess_adata(study_name, study)
            save_adata(adata_real_all, adata_pred_all, adata_real, adata_pred)
            #adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)
            ## Plot umap
            os.makedirs(f'figures/{study}/embedding', exist_ok=True)
            adata_real_all = create_umap_plot(adata_real_all, value_type="real", dataname=dataname, leiden_key="leiden_real", suffix="_all")
            adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_pred", suffix="_all")
            adata_real     = create_umap_plot(adata_real,     value_type="real", dataname=dataname, leiden_key="leiden_real")
            adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_pred")
            transfer_real_info(adata_real_all, adata_pred_all)
            transfer_real_info(adata_real,     adata_pred)
            adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_real", suffix="_all")
            adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_real")
            ## cluster score
            clust_stats_summary = calculate_cluster_stats(adata_pred_all, adata_pred, dataname, study)
            plot_barplot(clust_stats_summary, study)


## heatmap for marker genes
for pretrained_model in pretrained_models:
    for study_name, dataname in zip(studies, datanames):
        study = f'{study_name}__{pretrained_model}_{study_suffix}'
        print(study)
        try:
            df_corr = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
            cor_genes = df_corr.query('Correlation > 0.4')["Gene"].to_list()
            adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)
            curate_rank_genes(adata_real_all, "real") 
            curate_rank_genes(adata_pred_all, "pred") 
            curate_rank_genes(adata_real, "real") 
            curate_rank_genes(adata_pred, "pred") 
            plot_heatmap(adata_real_all, study, "real", "real", suffix="_all")
            plot_heatmap(adata_pred_all, study, "pred", "pred", suffix="_all")
            plot_heatmap(adata_real, study, "real", "real")
            plot_heatmap(adata_pred, study, "pred", "pred")
            adata_pred_all.uns["rank_genes_groups_curated_real"] = adata_real_all.uns["rank_genes_groups_curated_real"]
            adata_pred.uns["rank_genes_groups_curated_real"]     = adata_real.uns["rank_genes_groups_curated_real"]
            transfer_real_info(adata_real_all, adata_pred_all)
            transfer_real_info(adata_real,     adata_pred)
            plot_heatmap(adata_pred_all, study, "pred", "real", suffix="_all")
            plot_heatmap(adata_pred,     study, "pred", "real")
        except:
            continue


def merge_png(image_paths_all, image_paths_test, study):
    images_all = [Image.open(path) for path in image_paths_all]
    images_test = [Image.open(path) for path in image_paths_test]
    total_width = 1100 * 4
    total_height = 800 * 2
    new_image = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    for i, images in enumerate([images_all, images_test]):
        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, int(800 * i)))
            x_offset += 1100
    new_image.save(f"across_study/compare_embedding/umap/{study}/umap.png")



for study_name in studies:
    study = f'{study_name}__{study_suffix}'
    os.makedirs(f'across_study/compare_embedding/umap/{study}', exist_ok=True)
    image_paths_all = [
        f"figures/{study}/embedding/real_leiden_real_all.png"
    ] + [
        f"figures/{study_name}__{study_suffix}/embedding/pred_leiden_real_all.png" for study_suffix in study_suffixes]
    image_paths_test = [
        f"figures/{study}/embedding/real_leiden_real.png"
    ] + [
        f"figures/{study_name}__{study_suffix}/embedding/pred_leiden_real.png" for study_suffix in study_suffixes]
    merge_png(image_paths_all, image_paths_test, study)


## ondata
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman_K562_CRISPRa"
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'
adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)
adata_real_all = create_umap_plot(adata_real_all, value_type="real", dataname=dataname, leiden_key="leiden_real", suffix="_all", on_text=True)
adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_pred", suffix="_all", on_text=True)
adata_real     = create_umap_plot(adata_real,     value_type="real", dataname=dataname, leiden_key="leiden_real", on_text=True)
adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_pred", on_text=True)
transfer_real_info(adata_real_all, adata_pred_all)
transfer_real_info(adata_real,     adata_pred)
adata_pred_all = create_umap_plot(adata_pred_all, value_type="pred", dataname=dataname, leiden_key="leiden_real", suffix="_all", on_text=True)
adata_pred     = create_umap_plot(adata_pred,     value_type="pred", dataname=dataname, leiden_key="leiden_real", on_text=True)



## gene signature plot for Norman
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman_K562_CRISPRa"
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'
adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)

#https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html
def plot_program(adata, value_type, suffix=""):
    perturbation_list = {
        "Erythroid" : ['CBL_CNN1', 'CBL_PTPN12', 'CBL_PTPN9', 'CBL_UBASH3B', 'SAMD1_PTPN12', 
            'SAMD1_UBASH3B', 'UBASH3B_CNN1', 'UBASH3B_PTPN12', 'UBASH3B_PTPN9', 'UBASH3B_UBASH3A', 
            'UBASH3B_ZBTB25', 'BPGM_SAMD1', 'PTPN1', 'PTPN12_PTPN9', 'PTPN12_UBASH3A', 'PTPN12_ZBTB25'],
        "Granulocyte" : ['SPI1', 'CEBPA', 'CEBPB', 'CEBPE_CEBPA', 'CEBPE_RUNX1T1', 'CEBPE_SPI1', 
            'CEBPE', 'ETS2_CEBPE', 'KLF1_CEBPA', 'FOSB_CEBPE'],
        "Megakaryocyte" : ['MAPK1_TGFBR2', 'MAPK1', 'ETS2_MAPK1', 'ETS2', 'CEBPB_MAPK1'],
        "Pioneer_factor" : ['FOXA1_FOXF1', 'FOXA1_FOXL2', 'FOXA1_HOXB9', 'FOXA3_FOXA1', 
            'FOXA3_FOXF1', 'FOXA3_FOXL2', 'FOXA3_HOXB9', 'FOXA3', 'FOXF1_FOXL2', 'FOXF1_HOXB9', 'FOXL2_MEIS1', 
            'HOXA13', 'HOXC13', 'POU3F2_FOXL2', 'TP73', 'MIDN', 'LYL1_IER5L', 'DUSP9_SNAI1', 'ZBTB10_SNAI1'],
        "Pro_growth" : ['CEBPE_KLF1', 'KLF1', 'KLF1_BAK1', 'KLF1_TGFBR2', 'ELMSAN1', 'MAP2K3_SLC38A2', 
            'MAP2K3_ELMSAN1', 'MAP2K3', 'MAP2K3_MAP2K6', 'MAP2K6_ELMSAN1', 'MAP2K6', 'KLF1_MAP2K6'],
        "G1_cycle" : ['CDKN1A', 'CDKN1B_CDKN1A', 'CDKN1B', 'CDKN1C_CDKN1A', 'CDKN1C'],
    }
    programs = pd.DataFrame()
    for i,j in perturbation_list.items():
        program = pd.DataFrame({"Perturbation":j})
        program["Program"] = i
        programs = pd.concat([programs, program])
    adata.obs = pd.merge(adata.obs, programs, on="Perturbation", how="left")
    adata.obs["Program"] = adata.obs["Program"].fillna("others")
    sc.set_figure_params(fontsize=6, dpi=300, dpi_save=300)
    figsize=(6.2/2.54, 3.8/2.54)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sc.pl.umap(adata, color="Program", s=30, groups=list(perturbation_list.keys()), ax=ax)
    ax.legend(scatterpoints=1, markerscale=0.2, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)
    plt.tight_layout()
    plt.savefig(f'figures/{study}/gene_signature/{value_type}_program{suffix}.svg')
    plt.clf()
    plt.close()


def plot_signature(adata, signature, value_type):
    signature_list = {
        "Erythroid" : ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "GYPA", "ERMAP"],
        "Granulocyte" : ["ITGAM", "CSF3R", "LST1", "CD33"],
        "Megakaryocyte" : ["ITGA2B"]
        }
    for i,j in signature_list.items():
        sc.tl.score_genes(adata, j, score_name=i)
    signature_num = len(signature_list.keys())
    sc.set_figure_params(fontsize=6, dpi=300, dpi_save=300)
    fig, axes = plt.subplots(1, signature_num, figsize=(4.4/2.54 * signature_num, 3.8/2.54), dpi=300)
    for i, score in enumerate(signature_list.keys()):
        maxabs = max(abs(adata.obs[score]))
        sc.pl.umap(adata, color=score, cmap="RdBu_r", s=30, vmin=-maxabs, vmax=maxabs, vcenter=0, ax=axes[i], legend_fontsize="xx-small")
    plt.tight_layout()
    plt.savefig(f'figures/{study}/gene_signature/{value_type}_{signature}.svg')
    plt.clf()
    plt.close()


os.makedirs(f'figures/{study}/gene_signature', exist_ok=True)
plot_program(adata_real_all, "real", suffix="_all")
plot_program(adata_pred_all, "pred", suffix="_all")
plot_program(adata_real, "real")
plot_program(adata_pred, "pred")

plot_signature(adata_real_all, "immune_differentiation", "real")
plot_signature(adata_pred_all, "immune_differentiation", "pred")






## tf activity
#https://decoupler-py.readthedocs.io/en/latest/notebooks/dorothea.html#Activity-inference-with-univariate-linear-model-(ULM)
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman_K562_CRISPRa"
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'
adata_real_all, adata_pred_all, adata_real, adata_pred = load_adata(study)



def load_dataset_for_tfa():
    net = dc.get_collectri(organism='human', split_complexes=False)
    target_num = net.groupby("source")["target"].count().reset_index()
    net = net[net["source"].isin(target_num.query('target > 50')["source"].to_list())]
    tf_summary = pd.read_csv(f'reference/tf_summary/{study_name.replace("_train", "")}_tf_summary.txt', sep="\t")
    tf_summary["flag"] = 1
    return net, tf_summary

net, tf_summary = load_dataset_for_tfa()



def cal_tf_activity(_adata, net, tf_summary, value_type="real", dataset="all"):
    adata = _adata.copy()
    adata.X = adata.layers["scaled"]
    dc.run_ulm(mat=adata, net=net, source='source', target='target', weight='weight', use_raw=False)
    acts = dc.get_acts(adata, obsm_key='ulm_estimate')
    tf_summary = tf_summary[tf_summary["Pert"].isin(adata.obsm["ulm_estimate"].columns.to_list())]
    tf_pivot = tf_summary.pivot_table(index='Perturbation', columns='Pert', values='flag').fillna(0)
    score = adata.obsm["ulm_estimate"]
    obs_ncol = adata.obs.shape[1]
    adata.obs = adata.obs.merge(tf_pivot, left_index=True, right_index=True, how='left')
    adata.obs.iloc[:, obs_ncol:] = adata.obs.iloc[:, obs_ncol:].fillna(0)
    ranking = pd.DataFrame()
    for i in tf_pivot.columns:
        score_order = score[i].sort_values(ascending=False).index.to_list()
        sort_true = adata.obs.loc[score_order, [i]]
        sort_true["rank"] = [i for i in range(1, len(sort_true)+1)]
        sort_true = sort_true[sort_true[i] == 1]
        sort_true["TFs"] = i
        sort_true["value_type"] = value_type
        sort_true["dataset"] = dataset
        sort_true = sort_true.drop(i, axis=1)
        ranking = pd.concat([ranking, sort_true])
    return acts, ranking


summary_ranking = pd.DataFrame()
acts_real, ranking = cal_tf_activity(adata_real_all, net, tf_summary, value_type="real", dataset="all")
summary_ranking = pd.concat([summary_ranking, ranking])
acts_pred, ranking = cal_tf_activity(adata_pred_all, net, tf_summary, value_type="pred", dataset="all")
summary_ranking = pd.concat([summary_ranking, ranking])
#acts, ranking = cal_tf_activity(adata_real, net, tf_summary, value_type="real", dataset="test")
#summary_ranking = pd.concat([summary_ranking, ranking])
#acts, ranking = cal_tf_activity(adata_pred, net, tf_summary, value_type="pred", dataset="test")
#summary_ranking = pd.concat([summary_ranking, ranking])


def plot_stripplot(summary_ranking, study, ymin):
    order = summary_ranking.query('value_type == "real"').groupby("TFs")["rank"].mean().sort_values().index.to_list()
    plt.figure(figsize=(7.5/2.54, 5/2.54), dpi=300)
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.titlesize"] = 5
    plt.rcParams["axes.labelsize"] = 5
    sns.stripplot(data=summary_ranking, x="TFs", y="rank", hue="value_type", s=4, dodge=True, order=order, hue_order=["real", "pred"])
    plt.xlabel("TFs")
    plt.ylabel("Rank")
    plt.ylim(ymin, 0)
    plt.yticks([1, 25, 50, 75, 100, 125, 150, 175, 200, 220])
    plt.xticks(rotation=60)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"figures/{study}/TFactivity/ranking_tf_activity.svg")
    plt.clf()
    plt.close()


os.makedirs(f'figures/{study}/TFactivity', exist_ok=True)

plot_stripplot(summary_ranking, study, adata_real_all.shape[0])
summary_ranking.to_csv(f'figures/{study}/TFactivity/summary_ranking.txt', sep="\t")


def plot_umap_tf_activity(acts, value_type, summary_ranking):
    for i in set(summary_ranking["TFs"]):
        pert_list = summary_ranking.query('value_type == "real" and TFs == @i').index.to_list()
        sc.set_figure_params(fontsize=6, dpi=300, dpi_save=300)
        fig, ax = plt.subplots(figsize=(4.3/2.54, 3.8/2.54), dpi=300)
        sc.pl.umap(acts, color=[i], size=30, cmap='PiYG_r', vcenter=0, ax=ax, title=f"TF activity ({value_type})\n{i}")
        for j in acts.obs.reset_index().query('index == @pert_list').index.to_list():
            ax.text(acts.obsm["X_umap"][j][0], acts.obsm["X_umap"][j][1], acts.obs["Perturbation"][j], fontdict={"size": 4}, ha="left", va="top")
        adjust_text(ax.texts, arrowprops=dict(arrowstyle="-", color="black"))
        plt.tight_layout()
        plt.savefig(f"figures/{study}/TFactivity/{value_type}_umap_{i}.svg")
        plt.clf() 
        plt.close()


plot_umap_tf_activity(acts_real, "real", summary_ranking)
plot_umap_tf_activity(acts_pred, "pred", summary_ranking)








