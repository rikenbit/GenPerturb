#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack

try:
    import mudata as md
except ImportError as e:  # pragma: no cover
    raise SystemExit("mudata not installed; run with the singlecell env") from e

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("_common_11a", Path(__file__).with_name("11a_common.py"))
_common = module_from_spec(_spec)
_spec.loader.exec_module(_common)  # type: ignore[union-attr]
DATA_DIR = _common.DATA_DIR

DEST = DATA_DIR / "GSE288996"
OUT = DATA_DIR / "adata"
PREFIX = "Shevade2025_K562_DMSO"

# (sample_tag, RNA-GSM (multimodal mtx), ATAC-GSM (for fragments))
SAMPLES_DMSO = [
    ("K562_DMSO_1", "GSM8780566", "GSM8780558"),
    ("K562_DMSO_2", "GSM8780567", "GSM8780559"),
]


def _prefix(tag: str, gsm: str) -> Path:
    # tag "K562_DMSO_1" → sample "K562_DMSO", rep "1";
    # GEO filenames are "{gsm}_{sample}_RNA_{rep}_{suffix}"
    parts = tag.split("_")
    sample = "_".join(parts[:-1])
    rep = parts[-1]
    return DEST / f"{gsm}_{sample}_RNA_{rep}"


def load_multiome_mtx(tag: str, rna_gsm: str) -> ad.AnnData:
    pfx = _prefix(tag, rna_gsm)
    bc_file = Path(str(pfx) + "_barcodes.tsv.gz")
    ft_file = Path(str(pfx) + "_features.tsv.gz")
    mx_file = Path(str(pfx) + "_matrix.mtx.gz")
    for f in (bc_file, ft_file, mx_file):
        if not f.exists():
            raise FileNotFoundError(f)
    mtx = mmread(str(mx_file))
    X = csr_matrix(mtx).T  # cells x features
    barcodes = pd.read_csv(bc_file, header=None, sep="\t")[0].tolist()
    feats = pd.read_csv(ft_file, header=None, sep="\t",
                        names=["id", "symbol", "kind", "chrom", "start", "end"])
    feats["kind"] = feats["kind"].fillna("Gene Expression")
    var = feats.set_index("id")
    obs = pd.DataFrame(index=barcodes)
    return ad.AnnData(X=X, obs=obs, var=var)


def split_modalities(combined: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    is_gene = combined.var["kind"].str.startswith("Gene").values
    rna = combined[:, is_gene].copy()
    atac = combined[:, ~is_gene].copy()
    if {"chrom", "start", "end"}.issubset(atac.var.columns):
        atac.var_names = (
            atac.var["chrom"].astype(str) + ":"
            + atac.var["start"].astype(str) + "-"
            + atac.var["end"].astype(str)
        )
    return rna, atac


def load_guide_summary(tag: str) -> pd.DataFrame | None:
    # tag "K562_DMSO_1" → find "GSM*_K562_DMSO_guideRNA_1.txt.gz"
    parts = tag.split("_")
    sample = "_".join(parts[:-1])
    rep = parts[-1]
    hits = sorted(DEST.glob(f"GSM*_{sample}_guideRNA_{rep}.txt.gz"))
    if not hits:
        return None
    rows = []
    for f in hits:
        with gzip.open(f, "rt") as g:
            for ln in g:
                ln = ln.strip()
                if not ln:
                    continue
                toks = ln.split()
                if "=" in toks[0]:
                    guide, cnt = toks[0].split("=")[0], int(toks[1])
                else:
                    cnt, guide = int(toks[0]), toks[1]
                rows.append({"file": f.name, "guide": guide, "count": cnt})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-fragments", action="store_true",
                    help="stash ATAC fragments path into uns if the files exist")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    rnas, atacs, guide_rows = [], [], []
    for tag, rna_gsm, atac_gsm in SAMPLES_DMSO:
        try:
            combined = load_multiome_mtx(tag, rna_gsm)
        except FileNotFoundError as e:
            print(f"[skip] {tag}: missing {e}")
            continue
        rna, atac = split_modalities(combined)
        rna.obs["sample"] = tag
        atac.obs["sample"] = tag
        rna.obs["cell_type"] = "K562"
        atac.obs["cell_type"] = "K562"
        rna.obs["condition"] = "DMSO"
        atac.obs["condition"] = "DMSO"
        rna.obs_names = [f"{tag}_{b}" for b in rna.obs_names]
        atac.obs_names = [f"{tag}_{b}" for b in atac.obs_names]
        if args.with_fragments:
            parts = tag.split("_")
            sample = "_".join(parts[:-1])
            rep = parts[-1]
            frags = DEST / f"{atac_gsm}_{sample}_ATAC_{rep}_atac_fragments.tsv.gz"
            if frags.exists():
                atac.uns.setdefault("fragments_paths", {})[tag] = str(frags.resolve())
        rnas.append(rna)
        atacs.append(atac)
        gsum = load_guide_summary(tag)
        if gsum is not None:
            gsum["sample"] = tag
            guide_rows.append(gsum)
        print(f"[load] {tag}  rna={rna.shape}  atac={atac.shape}")

    if not rnas:
        print("[err ] no samples loaded; run 11a_fetch_Shevade2025.py first.")
        return 1

    rna_all = ad.concat(rnas, join="inner", merge="same")
    # ATAC peaks differ per replicate — take the union to keep both rep peaks.
    atac_all = ad.concat(atacs, join="outer", fill_value=0)
    print(f"[cat ] rna={rna_all.shape}  atac={atac_all.shape}")

    mu = md.MuData({"rna": rna_all, "atac": atac_all})
    mu.uns["source"] = "GSE288996 K562 DMSO; Shevade, Yang et al., bioRxiv 2025"
    if guide_rows:
        guide_df = pd.concat(guide_rows, ignore_index=True)
        mu.uns["guide_library_counts"] = guide_df.to_dict(orient="list")

    p_mu = OUT / f"{PREFIX}.h5mu"
    p_rna = OUT / f"{PREFIX}_rna.h5ad"
    p_atac = OUT / f"{PREFIX}_atac.h5ad"
    mu.write(str(p_mu))
    rna_all.write(str(p_rna))
    atac_all.write(str(p_atac))
    print(f"[save] {p_mu}")
    print(f"[save] {p_rna}")
    print(f"[save] {p_atac}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
