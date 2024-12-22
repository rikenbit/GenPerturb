# conda activate singlecell
import pertpy as pt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import gc
import os

def highly_variable_bygrna(adata, condition='gene', control='non-targeting', gene_col='ensembl_id'):
    hgenes = set()
    for i in set(adata.obs[condition]) - {control}:
        print(i)
        try:
            adata_sub = adata[adata.obs[condition].isin([i, control]), :].copy()
            sc.pp.highly_variable_genes(adata_sub, min_disp=0.2)
            hgene = set(adata_sub[:, adata_sub.var["highly_variable"] == True].var[gene_col])
            hgenes.update(hgene)
            del adata_sub
            gc.collect()
        except:
            continue
    adata.var["highly_variable"] = adata.var[gene_col].isin(hgenes)
    return adata

def process_and_write_data(adata_mix, prefix, control, perturbation_type='KD'):
    sc.pp.normalize_total(adata_mix, target_sum=1e4)
    sc.pp.log1p(adata_mix)
    adata_mix = highly_variable_bygrna(adata_mix)
    adata_mix = adata_mix[:, adata_mix.var["highly_variable"] == True]
    adata_mix.obs["perturbation"] = "Perturbed"
    adata_mix.obs.loc[adata_mix.obs["gene"] == control, "perturbation"] = "NT"
    adata_mix.obs["gene"] = list(adata_mix.obs["gene"])
    adata_mix.obs.loc[adata_mix.obs["gene"] == control, "gene"] = "NT"
    adata_mix.write(prefix + f'_mixscape_tmp_pre.h5ad')
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(adata_mix, 'perturbation', 'NT')
    adata_mix.layers["X_pert"] = sp.sparse.csr_matrix(adata_mix.layers["X_pert"])
    mixscape_identifier.mixscape(adata=adata_mix, control='NT', labels='gene', perturbation_type=perturbation_type, layer='X_pert')
    adata_mix.write(prefix + f'_mixscape_tmp.h5ad')
    return adata_mix

def filter_adata(adata_mix, _adata, prefix):
    _adata.obs = adata_mix.obs
    hgene = list(adata_mix.var["ensembl_id"])
    _adata.var["highly_variable"] = _adata.var["ensembl_id"].isin(hgene)
    _adata.write(prefix + '_mixscape.h5ad')
    _adata = _adata[_adata.obs["mixscape_class_global"] != "NP"]
    _adata.write(prefix + '_mixscape_exnp.h5ad')

# Process the data
input_files = [
    "data/adata/ReplogleWeissman2022_K562_gwps_1.h5ad",
    "data/adata/ReplogleWeissman2022_K562_gwps_2.h5ad",
    "data/adata/ReplogleWeissman2022_K562_gwps_3.h5ad",
    "data/adata/ReplogleWeissman2022_K562_gwps_4.h5ad",
    "data/adata/ReplogleWeissman2022_K562_gwps_5.h5ad",
    "data/adata/ReplogleWeissman2022_K562_gwps_6.h5ad",
    "data/adata/ReplogleWeissman2022_K562_essential.h5ad",
    "data/adata/ReplogleWeissman2022_rpe1.h5ad"
]

control = "non-targeting"
for input_adata in input_files:
    adata = sc.read_h5ad(input_adata)
    prefix = input_adata.replace(".h5ad", "")
    adata_mix = adata.copy()
    adata_mix = process_and_write_data(adata_mix, prefix, control, perturbation_type='KD')
    filter_adata(adata_mix, adata, prefix)


input_adata = "data/adata/NormanWeissman2019_filtered.h5ad"
control = "control"
adata = sc.read_h5ad(input_adata)
adata.obs.rename(columns={"perturbation":"gene"}, inplace=True)
adata.var.rename(columns={"ensemble_id":"ensembl_id"}, inplace=True)
prefix = input_adata.replace(".h5ad", "")
adata_mix = adata.copy()
adata_mix = process_and_write_data(adata_mix, prefix, control, perturbation_type='OE')
filter_adata(adata_mix, adata, prefix)
