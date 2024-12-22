## conda activate singlecell
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from adpbulk import ADPBulk

def process_and_save_data(adata_input, col, pert, ctrl, pert_num, study, save=True):
    adata = adata_input[adata_input.obs[col].isin(adata_input.obs.groupby([col]).count().query(f'{adata_input.obs.columns[3]} > 100').index.tolist())].copy()
    adata.obs = adata.obs.rename(columns={col: pert})
    adpb = ADPBulk(adata, pert, name_delim="_", method="sum")
    df = adpb.fit_transform().T
    df = df * (1000000 / df.sum())
    ctrl_col = df.pop(f"{pert}.{ctrl}")
    df.insert(0, f"{pert}.{ctrl}", ctrl_col)
    df = df.copy()
    if pert_num >= 1:
        df = df.T
        genes = ["Gene1", "Gene2"] if pert_num == 2 else ["Gene"]
        id_map = adata.obs.loc[:,[pert] + genes].drop_duplicates().set_index(pert)
        id_map.index = pert + "." + id_map.index.astype("str")
        df = pd.merge(df, id_map, left_index=True, right_index=True)
        for column in genes:
            for gene in df[column].to_list():
                try:
                    df.loc[df[column] == gene, gene] = df.loc[f"{pert}.{ctrl}", gene]
                except:
                    continue
        df = df.drop(genes, axis=1).T
    if save:
        df.astype("float32").to_csv(f'data/{study}_cpm.tsv', sep="\t")
    else:
        return df.astype("float32"), fc.astype("float32")

### Norman ###
study = "NormanWeissman2019_filtered_mixscape_exnp"
pert = "Norman"
col  = "gene"
ctrl = "NT"
pert_num = 2
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
adata.obs.loc[:, ["Gene1", "Gene2"]] = pd.DataFrame([i.split("_") for i in adata.obs["gene"]]).values
process_and_save_data(adata, col, pert, ctrl, pert_num, study)

### Replogle ###
## gwps ##
study = "ReplogleWeissman2022_K562_gwps_mixscape_exnp"
pert = "Replogle_gwps"
col  = "gene"
ctrl = "NT"
pert_num = 1
adatas = []
for i in range(1,7):
    adatap = sc.read_h5ad(f"data/adata/ReplogleWeissman2022_K562_gwps_{i}_mixscape_exnp.h5ad")
    if i != 1:
        adatap = adatap[adatap.obs["gene"] != ctrl]
    adatas.append(adatap)


adata = ad.concat(adatas)
adata.obs.loc[:, ["Gene"]] = adata.obs["gene"]
process_and_save_data(adata, col, pert, ctrl, pert_num, study)

## gwps essential ##
study = "ReplogleWeissman2022_K562_essential_mixscape_exnp"
pert = "Replogle_essential"
col  = "gene"
ctrl = "NT"
pert_num = 1
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
adata.obs.loc[:, ["Gene"]] = adata.obs["gene"]
process_and_save_data(adata, col, pert, ctrl, pert_num, study)

## gwps rpe1 ##
study = "ReplogleWeissman2022_rpe1_mixscape_exnp"
pert = "Replogle_rpe1"
col  = "gene"
ctrl = "NT"
pert_num = 1
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
adata.obs.loc[:, ["Gene"]] = adata.obs["gene"]
process_and_save_data(adata, col, pert, ctrl, pert_num, study)


### Srivatsan ###
## common ##
col  = "ids"
ctrl = "Vehicle_Vehicle_0nM"
pert_num = 0

## K562 ##
study = f"Srivatsan2019_K562"
pert = "Srivatsan_K562"
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
process_and_save_data(adata, col, pert, ctrl, pert_num, study)
 
## MCF7 ##
study = f"Srivatsan2019_MCF7"
pert = "Srivatsan_MCF7"
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
process_and_save_data(adata, col, pert, ctrl, pert_num, study)

## A549 ##
study = f"Srivatsan2019_A549"
pert = "Srivatsan_A549"
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
process_and_save_data(adata, col, pert, ctrl, pert_num, study)


### TF atlas ###
study = "JuliaJoung2023_TFAtlas"
pert = "Julia"
col = "gene"
ctrl = "NT"
pert_num = 2
adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
adata.var["ensembl_id"] = adata.var.index
process_and_save_data(adata, col, pert, ctrl, pert_num, study)



### D-SPIN ###

for j in ["B_cell", "CD4T", "CD8T", "Myeloid"]:
    study = f"JialongJiang2024_{j}"
    pert = "Jialong"
    col  = "sample_id"
    ctrl = "CONTROL_CD3"
    pert_num = 0
    adata = sc.read_h5ad(f"data/adata/{study}.h5ad")
    process_and_save_data(adata, col, pert, ctrl, pert_num, study)



