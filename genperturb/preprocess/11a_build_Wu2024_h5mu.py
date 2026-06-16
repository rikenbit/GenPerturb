#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

import mudata as md

try:
    md.set_options(pull_on_update=False)
except AttributeError:
    pass

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("_common_11a", Path(__file__).with_name("11a_common.py"))
_common = module_from_spec(_spec)
_spec.loader.exec_module(_common)  # type: ignore[union-attr]
DATA_DIR = _common.DATA_DIR

SRC = DATA_DIR / "GSE277747" / "converted"
OUT = DATA_DIR / "adata"
PREFIX = "Wu2024"


def _load_modality(tag: str) -> ad.AnnData:
    mtx = mmread(str(SRC / f"{tag}.mtx"))
    X = csr_matrix(mtx).T  # cells x features
    barcodes = pd.read_csv(SRC / f"{tag}_barcodes.tsv",
                           header=None, sep="\t")[0].tolist()
    feats = pd.read_csv(SRC / f"{tag}_features.tsv", sep="\t")
    feats = feats.set_index("rowname", drop=False)
    obs = pd.DataFrame(index=barcodes)
    return ad.AnnData(X=X, obs=obs, var=feats)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] RNA ...")
    rna = _load_modality("rna")
    rna.var_names = rna.var["ensembl_gene"].astype(str)
    rna.var_names.name = None
    rna.var["species"] = np.where(
        rna.var["ensembl_gene"].str.startswith("ENSMUS"), "mouse", "human"
    )
    print(f"       {rna.shape}  var cols: {list(rna.var.columns)}")

    print("[load] ATAC ...")
    atac = _load_modality("atac")
    atac.var_names = atac.var["concat"].astype(str)
    atac.var_names.name = None
    start_i = pd.to_numeric(atac.var["peak_start"], errors="coerce").astype("Int64")
    end_i = pd.to_numeric(atac.var["peak_end"], errors="coerce").astype("Int64")
    atac.var["peak_coord"] = (
        atac.var["peak_chr"].astype(str) + ":"
        + start_i.astype(str) + "-" + end_i.astype(str)
    )
    print(f"       {atac.shape}  var cols: {list(atac.var.columns)}")

    print("[load] colData (guide assignments) ...")
    cells = pd.read_csv(SRC / "cells.tsv", sep="\t").set_index("barcode")
    print(f"       {len(cells)} cells, cols: {list(cells.columns)}")

    for a in (rna, atac):
        a.obs = a.obs.join(cells)
        a.obs["is_NT"] = a.obs["guide_gene"].astype(str).str.contains(
            "non-targeting", case=False, na=False)
        a.obs["is_mouse_NT"] = a.obs["guide_gene"].astype(str).str.contains(
            "mouse", case=False, na=False)

    mu = md.MuData({"rna": rna, "atac": atac})
    mu.uns["source"] = "GSE277747 GSM8528725; Wu et al., Nat Biotechnol 2024"
    mu.uns["guide_gene_counts"] = (
        cells["guide_gene"].value_counts().to_dict()
    )

    if not args.no_write:
        p_mu = OUT / f"{PREFIX}.h5mu"
        p_rna = OUT / f"{PREFIX}_rna.h5ad"
        p_atac = OUT / f"{PREFIX}_atac.h5ad"
        mu.write(str(p_mu))
        rna.write(str(p_rna))
        atac.write(str(p_atac))
        print(f"[save] {p_mu}")
        print(f"[save] {p_rna}")
        print(f"[save] {p_atac}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
