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


## TF atlas ##
input_adata = "data/adata/GSE217460_210322_TFAtlas.h5ad"
control = "NT"
adata = sc.read_h5ad(input_adata)
adata.X = (adata.X.T  * np.array(adata.obs["n_counts"]) / 10000).T
adata.X = adata.X.round(0)
adata.X = sp.sparse.csr_matrix(adata.X)

adata.var["gene"] = adata.var.index
adata.obs.TF = adata.obs.TF.astype("str")
adata.obs.loc[adata.obs.TF.str.contains("GFP"), "TF"] = control
adata.obs.loc[adata.obs.TF.str.contains("mCherry"), "TF"] = control

adata.obs["Gene"] = [i.split("-", 1)[-1] for i in adata.obs["TF"]]
adata.obs["Gene"] = [i.replace("-", ",") if i in ["ZNF559-ZNF177", "BORCS8-MEF2B", "CCDC169-SOHLH2"] else i for i in adata.obs["Gene"]]
adata.obs.loc[:, ["Gene1", "Gene2"]] = pd.DataFrame([i.split(",") for i in adata.obs["Gene"]]).values
adata = adata[adata.obs["TF"].isin(adata.obs.groupby(["TF"]).count().query('batch > 30').index.tolist())]
adata.obs.rename(columns={"TF":"gene"}, inplace=True)
adata.write('data/adata/JuliaJoung2023_TFAtlas.h5ad')


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






