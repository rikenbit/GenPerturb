## conda activate singlecell
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from adpbulk import ADPBulk


def process_and_save_atac(adata_input, col, pert, ctrl, study,
                          keep_barcodes=None):
    if keep_barcodes is not None:
        n_before = adata_input.shape[0]
        mask = adata_input.obs.index.isin(keep_barcodes)
        adata_input = adata_input[mask].copy()
        n_after = adata_input.shape[0]
        print(f"    Mixscape filter: {n_after:,}/{n_before:,} cells "
              f"({100*n_after/max(n_before,1):.1f}%) retained")
    adata = adata_input[adata_input.obs[col].isin(
        adata_input.obs.groupby([col]).count().query(
            f'{adata_input.obs.columns[3]} > 100').index.tolist()
    )].copy()
    adata.obs = adata.obs.rename(columns={col: pert})
    adpb = ADPBulk(adata, pert, name_delim="_", method="sum")
    df = adpb.fit_transform().T
    df = df * (1000000 / df.sum())
    ctrl_col = df.pop(f"{pert}.{ctrl}")
    df.insert(0, f"{pert}.{ctrl}", ctrl_col)
    df = df.copy()
    df.astype("float32").to_csv(f'data/{study}_atac_cpm.tsv', sep="\t")
    print(f"    wrote data/{study}_atac_cpm.tsv  shape={df.shape}")


def load_mixscape_exnp_barcodes(prefix):
    rna_path = f"{prefix}_mixscape_exnp.h5ad"
    rna = ad.read_h5ad(rna_path, backed="r")
    barcodes = set(rna.obs.index.tolist())
    print(f"  loaded {len(barcodes):,} exNP barcodes from {rna_path}")
    return barcodes


### MartinRufino ATAC ###
martin_keep = load_mixscape_exnp_barcodes("data/adata/MartinRufino2025")

adata = sc.read_h5ad("data/adata/MartinRufino2025_atac.h5ad")
study = "MartinRufino2025"
print(f"  {study}: {adata.shape[0]} cells (pre-filter)")
process_and_save_atac(adata, "target", "MartinRufino", "NT", study,
                       keep_barcodes=martin_keep)

## Per-cell-type datasets
celltype_safenames = [
    "ProErythroblast",
    "PolychromaticErythroblast",
    "BasophilicErythroblast",
    "OrthochromaticErythroblast",
    "EoBasoMastPrecursor",
    "CFUE",
]
for ct_safe in celltype_safenames:
    h5ad_path = f"data/adata/MartinRufino2025_{ct_safe}_atac.h5ad"
    try:
        adata_ct = sc.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"  Skipping {ct_safe}: h5ad not found")
        continue
    study = f"MartinRufino2025_{ct_safe}"
    print(f"  {study}: {adata_ct.shape[0]} cells (pre-filter)")
    process_and_save_atac(adata_ct, "target", "MartinRufino", "NT", study,
                          keep_barcodes=martin_keep)
