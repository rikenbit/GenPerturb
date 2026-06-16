# conda activate singlecell

import scanpy as sc
import scipy as sp
import numpy as np
import pandas as pd
import math


input_adata = "data/adata/ReplogleWeissman2022_K562_gwps_raw.h5ad"
adata = sc.read_h5ad(input_adata)
adata.X = sp.sparse.csr_matrix(adata.X)
adata.write('data/adata/ReplogleWeissman2022_K562_gwps.h5ad')


## Replogle ##
input_adata = "data/adata/ReplogleWeissman2022_K562_gwps.h5ad"
adata = sc.read_h5ad(input_adata)
adata.X = sp.sparse.csr_matrix(adata.X)
genes = list(set(adata.obs.gene) - set(["non-targeting"]))
pert_num = len(genes) // 6
genes1 = genes[:pert_num]
genes2 = genes[pert_num:pert_num * 2]
genes3 = genes[pert_num * 2:pert_num * 3]
genes4 = genes[pert_num * 3:pert_num * 4]
genes5 = genes[pert_num * 4:pert_num * 5]
genes6 = genes[pert_num * 5:]
adata[adata.obs["gene"].isin(genes1 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_1.h5ad')
adata[adata.obs["gene"].isin(genes2 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_2.h5ad')
adata[adata.obs["gene"].isin(genes3 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_3.h5ad')
adata[adata.obs["gene"].isin(genes4 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_4.h5ad')
adata[adata.obs["gene"].isin(genes5 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_5.h5ad')
adata[adata.obs["gene"].isin(genes6 + ["non-targeting"])].write('data/adata/ReplogleWeissman2022_K562_gwps_6.h5ad')



## Srivatsan ##
adata_all = sc.read_h5ad(f"data/adata/Srivatsan_2019_raw.h5ad")
adata_all.obs.loc[adata_all.obs['vehicle'] == 1, 'target'] = "Vehicle"
adata_all.obs["target"] = [i.replace(" ", "") for i in adata_all.obs["target"]]
adata_all.obs["product_dose"] = [i.replace(" ", "") + "nM" for i in adata_all.obs["product_dose"]]
#adata_all.obs["product_dose"] = [i.replace(" ", ",").replace("(", "").replace(")", "") for i in adata_all.obs["product_dose"]]
adata_all.obs["ids"] = adata_all.obs["target"] + "_" + adata_all.obs["product_dose"]
adata_all[adata_all.obs["cell_type"] == "K562"].write('data/adata/Srivatsan2019_K562.h5ad')
adata_all[adata_all.obs["cell_type"] == "MCF7"].write('data/adata/Srivatsan2019_MCF7.h5ad')
adata_all[adata_all.obs["cell_type"] == "A549"].write('data/adata/Srivatsan2019_A549.h5ad')



## DSPIN ##
adata = sc.read_h5ad("data/adata/drug_profiling_raw_counts.h5ad")
del adata.obsm
del adata.uns
adata = adata[~(adata.obs["cell_type_coarse"] == "Other")]
adata.obs["sample_id"] = ["CONTROL_CD3" if "CONTROL_CD3" in i else "CONTROL" if "CONTROL" in i else i  for i in adata.obs["sample_id"]]
adata.obs["group"] = adata.obs["sample_id"].astype("str") + "_" + adata.obs["cell_type_coarse"].astype("str")
adata.write("data/adata/JialongJiang2024.h5ad")

adata_sub = adata[adata.obs["CD3"] == 1]
for j in set(adata_sub.obs["cell_type_coarse"]) - set(["NK"]):
    adata_sub2 = adata_sub[adata_sub.obs["cell_type_coarse"] == j]
    adata_sub2.write(f"data/adata/JialongJiang2024_{j}.h5ad")



## MartinRufino ##
import anndata as ad
import os

meta = pd.read_csv("data/MartinRufino/GSE274113_annotated_metadata.csv", index_col=0)
meta = meta.dropna(how="all")

rep_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16]
adatas_rna = []
adatas_atac = []
for rep_id in rep_ids:
    h5_path = f"data/MartinRufino/GSE274113_rep{rep_id}_filtered_feature_bc_matrix.h5"
    if not os.path.exists(h5_path):
        print(f"  Skipping {h5_path} (not found)")
        continue
    print(f"  Loading rep{rep_id}...")
    adata_rep = sc.read_10x_h5(h5_path, gex_only=False)
    adata_rep.X = sp.sparse.csr_matrix(adata_rep.X)
    adata_rep.obs_names = [f"rep{rep_id}_{bc}" for bc in adata_rep.obs_names]
    adata_rep.var_names_make_unique()
    is_rna = adata_rep.var["feature_types"] == "Gene Expression"
    is_atac = adata_rep.var["feature_types"] == "Peaks"
    adatas_rna.append(adata_rep[:, is_rna].copy())
    adatas_atac.append(adata_rep[:, is_atac].copy())
    del adata_rep

adata_rna = ad.concat(adatas_rna, merge="same")
del adatas_rna
adata_atac = ad.concat(adatas_atac, join="outer", fill_value=0)
del adatas_atac

common_cells = adata_rna.obs_names.intersection(meta.index)
adata_rna = adata_rna[common_cells].copy()
adata_atac = adata_atac[common_cells].copy()

for col in meta.columns:
    adata_rna.obs[col] = meta.loc[adata_rna.obs_names, col].values
    adata_atac.obs[col] = meta.loc[adata_atac.obs_names, col].values


os.makedirs("data/adata", exist_ok=True)
adata_rna.write("data/adata/MartinRufino2025.h5ad")
adata_atac.write("data/adata/MartinRufino2025_atac.h5ad")


## Wu2024 (GSE277747, MultiPerturb-seq) ##
wu_rna_path = "data/adata/Wu2024_rna.h5ad"
wu_atac_path = "data/adata/Wu2024_atac.h5ad"
if os.path.exists(wu_rna_path) and os.path.exists(wu_atac_path):
    print(f"  [Wu2024] filtering human cells + features...")
    adata_rna = sc.read_h5ad(wu_rna_path)
    adata_atac = sc.read_h5ad(wu_atac_path)
    rna_keep_var = adata_rna.var["species"].astype(str) == "human"
    adata_rna = adata_rna[:, rna_keep_var.values].copy()
    atac_keep_var = adata_atac.var["species"].astype(str) == "human"
    adata_atac = adata_atac[:, atac_keep_var.values].copy()
    g = adata_rna.obs["guide_gene"].astype(str).fillna("")
    keep = (~g.str.contains("mouse", case=False, na=False)) & (g != "") & (g != "nan")
    adata_rna = adata_rna[keep.values].copy()
    adata_atac = adata_atac[keep.values].copy()
    gg = adata_rna.obs["guide_gene"].astype(str).fillna("NT")
    gg = gg.str.replace(r"\s*\(human\)\s*$", "", regex=True)
    gg = gg.where(~gg.str.contains("non-targeting", case=False, na=False), "NT")
    for a in (adata_rna, adata_atac):
        a.obs["target"] = gg.values
    adata_rna.var["gene_ids"] = adata_rna.var["ensembl_gene"].astype(str)
    adata_atac.var["gene_ids"] = adata_atac.var_names.astype(str)
    adata_rna.X = sp.sparse.csr_matrix(adata_rna.X)
    adata_atac.X = sp.sparse.csr_matrix(adata_atac.X)
    adata_rna.write("data/adata/Wu2024.h5ad")
    adata_atac.write("data/adata/Wu2024_atac_filtered.h5ad")


## Metzner2025 (PRJNA1128171, Zenodo 10.5281/zenodo.15116138) ##
metzner_path = "data/PRJNA1128171/h5ad/multiome_perturb_seq.h5ad"
if os.path.exists(metzner_path):
    print(f"  [Metzner2025] preparing chromatin remodeller perturb-seq...")
    adata = sc.read_h5ad(metzner_path)
    target = adata.obs["guide_target"].astype(str)
    adata.obs["target"] = target.where(target != "NTC", "NT").values
    adata.var["gene_symbol"] = adata.var_names.astype(str)
    adata.var["gene_ids"] = adata.var["gene_ids"].astype(str)
    adata.var_names_make_unique()
    adata.X = sp.sparse.csr_matrix(adata.X)
    adata.write("data/adata/Metzner2025.h5ad")


## Shevade2025 (GSE288996, CAT-ATAC; K562 DMSO only) ##
shev_rna_path = "data/adata/Shevade2025_K562_DMSO_rna.h5ad"
if os.path.exists(shev_rna_path):
    print(f"  [Shevade2025] preparing K562 DMSO baseline...")
    adata = sc.read_h5ad(shev_rna_path)
    if "cell_type" in adata.obs:
        adata = adata[adata.obs["cell_type"].astype(str) == "K562"].copy()
    if "condition" in adata.obs:
        adata = adata[adata.obs["condition"].astype(str) == "DMSO"].copy()
    adata.obs["target"] = "NT"
    adata.X = sp.sparse.csr_matrix(adata.X)
    if "symbol" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["symbol"].astype(str)
        adata.var_names = np.where(
            adata.var["gene_symbol"].eq(""), adata.var_names, adata.var["gene_symbol"].values
        )
        adata.var_names_make_unique()
    adata.var["gene_ids"] = adata.var.index.astype(str)
    adata.write("data/adata/Shevade2025_K562_DMSO.h5ad")


