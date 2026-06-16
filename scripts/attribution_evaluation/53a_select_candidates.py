#!/usr/bin/env python
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CWD = Path(__file__).resolve().parents[2]
SCRIPT_DIR = CWD / "scripts" / "attribution_evaluation"

sys.path.insert(0, str(SCRIPT_DIR))
import importlib  # noqa: E402
atac_utils = importlib.import_module("53_atac_utils")  # noqa: E402

STUDY_DIRNAME = "Martin_full"
DATA_DIR = CWD / "attribution_analysis" / "insilico_mutation" / STUDY_DIRNAME
SEQLET_META = DATA_DIR / "seqlet_metadata.tsv"
SITE_TABLE = DATA_DIR / "tables" / "site_level_table.tsv"

ATAC_TSV = CWD / "data" / "MartinRufino2025_atac_cpm.tsv"
ATAC_CTRL_COL = "MartinRufino.NT"
TRAIN_BED = CWD / "fasta" / "MartinRufino2025_mixscape_exnp_train.bed"

OUT_DIR = DATA_DIR / "paper_figure"

MIN_ABS_FCWT = 0.10
MIN_CANCEL_PCT = 0.10
MIN_ATAC_DELTA = 0.30
ATAC_FLANK = 200
MIN_DIST_TO_TSS = 2000
TOP_PER_PERT = 10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading seqlet_metadata...")
    sm = pd.read_csv(SEQLET_META, sep="\t")
    print(f"  {len(sm)} seqlets")

    sm = sm[(sm["gene_match"] == True) | (sm["cluster_match"] == True)].copy()
    sm = sm.dropna(subset=["core_genomic_start"]).copy()
    sm["core_genomic_start"] = sm["core_genomic_start"].astype(int)
    sm["core_genomic_end"] = sm["core_genomic_end"].astype(int)
    sm["genomic_start"] = sm["genomic_start"].astype(int)
    sm["genomic_end"] = sm["genomic_end"].astype(int)
    print(f"  gene_match | cluster_match: {len(sm)} seqlets across "
          f"{sm['gene'].nunique()} genes and {sm['perturbation'].nunique()} perts")

    print("Loading site_level_table (seqlet_mut)...")
    sl = pd.read_csv(SITE_TABLE, sep="\t")
    sl = sl[sl["mutation_type"] == "seqlet_mut"].copy()
    print(f"  {len(sl):,} site rows")

    cand = sm.merge(
        sl[["gene", "perturbation", "site_id",
            "wt_ctrl", "wt_pert", "mt_ctrl", "mt_pert",
            "fc_wt", "fc_mt", "attr_sum_abs", "attr_max_abs",
            "delta_fc"]],
        on=["gene", "perturbation", "site_id"],
        how="inner",
        suffixes=("", "_sl"),
    )
    cand["abs_fc_wt"] = cand["fc_wt"].abs()
    cand["abs_fc_mt"] = cand["fc_mt"].abs()
    cand["cancel"] = cand["abs_fc_wt"] - cand["abs_fc_mt"]
    cand["cancel_pct"] = cand["cancel"] / cand["abs_fc_wt"].clip(lower=1e-6)
    print(f"  {len(cand)} candidates after merge")

    cand = cand[(cand["abs_fc_wt"] >= MIN_ABS_FCWT)
                & (cand["cancel_pct"] > MIN_CANCEL_PCT)].copy()
    print(f"  {len(cand)} after |fc_wt|>={MIN_ABS_FCWT} "
          f"and cancellation%>{MIN_CANCEL_PCT*100:.0f}%")

    print(f"Loading ATAC pseudobulk CPM table {ATAC_TSV.name} ...")
    atac_cols, atac_matrix, atac_by_chrom = atac_utils.load_atac_table(ATAC_TSV)
    print(f"  {atac_matrix.shape[0]:,} peaks × {atac_matrix.shape[1]} columns")

    print("Computing ATAC delta CPM around each seqlet ...")
    atac_n = np.zeros(len(cand), dtype=np.int32)
    atac_max = np.zeros(len(cand), dtype=np.float32)
    atac_sum = np.zeros(len(cand), dtype=np.float32)
    atac_signed = np.zeros(len(cand), dtype=np.float32)
    for i, (_, row) in enumerate(cand.iterrows()):
        s = atac_utils.seqlet_delta(
            atac_matrix, atac_by_chrom, atac_cols,
            row["chromosome"], int(row["core_genomic_start"]),
            int(row["core_genomic_end"]),
            ctrl_col=ATAC_CTRL_COL, pert_col=row["perturbation"],
            flank=ATAC_FLANK,
        )
        atac_n[i] = s["n_peaks"]
        atac_max[i] = s["max_abs_delta"]
        atac_sum[i] = s["sum_abs_delta"]
        atac_signed[i] = s["signed_delta"]
    cand["atac_n_peaks"] = atac_n
    cand["atac_max_abs_delta"] = atac_max
    cand["atac_sum_abs_delta"] = atac_sum
    cand["atac_signed_delta"] = atac_signed

    cand = cand[cand["atac_max_abs_delta"] >= MIN_ATAC_DELTA].copy()
    print(f"  {len(cand)} after ATAC max |delta CPM| >= {MIN_ATAC_DELTA}"
          f" (flank ±{ATAC_FLANK} bp)")

    print(f"Applying TSS-distal filter (|core_mid − TSS| > {MIN_DIST_TO_TSS} bp)...")
    bed = pd.read_csv(TRAIN_BED, sep="\t", header=None,
                      names=["chrom", "start", "end", "gene", "score",
                             "strand", "training"])
    tss_map = {r["gene"]: int(r["start"]) for _, r in bed.iterrows()}
    cand["tss_pos"] = cand["gene"].map(tss_map)
    cand = cand.dropna(subset=["tss_pos"]).copy()
    cand["tss_pos"] = cand["tss_pos"].astype(int)
    cand["core_mid"] = (
        (cand["core_genomic_start"] + cand["core_genomic_end"]) // 2
    ).astype(int)
    cand["dist_to_tss"] = (cand["core_mid"] - cand["tss_pos"]).abs().astype(int)
    cand = cand[cand["dist_to_tss"] > MIN_DIST_TO_TSS].copy()
    print(f"  {len(cand)} after distance filter "
          f"(median dist={int(cand['dist_to_tss'].median()) if len(cand) else 0} bp)")

    cand = cand.sort_values(["perturbation", "attr_sum_abs"], ascending=[True, False])
    parts = []
    for p, sub in cand.groupby("perturbation"):
        sub = sub.drop_duplicates(subset=["gene"]).head(TOP_PER_PERT)
        parts.append(sub)
    cand = pd.concat(parts).sort_values(["perturbation", "attr_sum_abs"],
                                         ascending=[True, False]).reset_index(drop=True)
    print(f"  {len(cand)} final (top {TOP_PER_PERT} per pert, dedup by gene)")
    print(cand["perturbation"].value_counts())

    keep_cols = [
        "perturbation", "pert_gene", "gene", "matched_motif_gene", "matched_motif",
        "matched_qval", "chromosome", "genomic_start", "genomic_end",
        "core_genomic_start", "core_genomic_end", "core_length",
        "is_revcomp", "jaspar_id", "motif_width",
        "site_id", "seqlet_instance_id",
        "wt_ctrl", "wt_pert", "mt_ctrl", "mt_pert",
        "fc_wt", "fc_mt", "abs_fc_wt", "abs_fc_mt", "cancel", "cancel_pct", "delta_fc",
        "attr_sum_abs", "attr_max_abs",
        "atac_n_peaks", "atac_max_abs_delta", "atac_sum_abs_delta", "atac_signed_delta",
        "tss_pos", "core_mid", "dist_to_tss",
    ]
    cand = cand[keep_cols]
    out_path = OUT_DIR / "candidates.tsv"
    cand.to_csv(out_path, sep="\t", index=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
