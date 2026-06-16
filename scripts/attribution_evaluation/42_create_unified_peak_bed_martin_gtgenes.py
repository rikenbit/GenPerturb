#!/usr/bin/env python
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pybedtools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bed_utils import (  # noqa: E402
    MODEL_CONTEXT_LENGTH,
    safe_makedirs,
    load_chrom_sizes,
    clip_gene_windows,
    filter_primary_chroms_df,
    ensure_bed3_safe_df,
    expand_to_tss_window,
    expand_to_promoter,
    filter_peaks_to_tss_window,
)
from _gene_aware_intervals import merge_intervals_by_gene  # noqa: E402



def load_re2g_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", header=None,
                     compression="gzip", low_memory=False)
    df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "name",
                            4: "class", 5: "TargetGene"})
    df["score"] = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    df["strand"] = "."
    df = filter_primary_chroms_df(df, chr_col="chr")
    df["gene"] = df["TargetGene"].astype(str).replace("NA", pd.NA)
    df = df.dropna(subset=["gene"]).copy()
    return df


def load_abc_data(path: str, cell_type: str) -> pd.DataFrame:
    print(f"[INFO] Loading ABC table {path} (cell_type={cell_type})")
    df = pd.read_csv(path, sep="\t", compression="gzip", header=None, low_memory=False)
    ncols = df.shape[1]
    df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "name",
                            4: "class", 6: "TargetGene"})
    df["cell_type"] = df.iloc[:, -1].astype(str)

    candidates = []
    for col_idx in range(max(15, ncols - 5), ncols - 1):
        col_data = pd.to_numeric(df.iloc[:, col_idx], errors="coerce")
        if col_data.notna().sum() > 0:
            cmin, cmax = col_data.min(), col_data.max()
            if 0 <= cmin and cmax <= 2:
                candidates.append((col_idx, col_data.mean()))
    if candidates:
        best = min(candidates, key=lambda x: abs(x[1] - 0.1))[0]
        df["ABC.Score"] = pd.to_numeric(df.iloc[:, best], errors="coerce")
    elif ncols > 20:
        df["ABC.Score"] = pd.to_numeric(df.iloc[:, 20], errors="coerce")
    else:
        df["ABC.Score"] = pd.to_numeric(df.iloc[:, -2], errors="coerce")

    df = df[df["cell_type"] == cell_type].copy()
    df = filter_primary_chroms_df(df, chr_col="chr")
    df["gene"] = df["TargetGene"].astype(str).replace("NA", pd.NA)
    df = df.dropna(subset=["gene", "ABC.Score"]).copy()
    print(f"[INFO] ABC rows for {cell_type}: {len(df)}")
    return df


def load_table_s3_genes(xlsx_path: str) -> Dict[str, set]:
    df = pd.read_excel(xlsx_path, sheet_name="TF_sensitive_genes", header=2)
    df = df.dropna(subset=["gene_ID", "perturbation_name"])
    out = {}
    for pert, group in df.groupby("perturbation_name"):
        out[str(pert)] = set(group["gene_ID"].astype(str).tolist())
    return out


def load_atac_bed(bed_path: str) -> pd.DataFrame:
    if not os.path.exists(bed_path):
        return pd.DataFrame(columns=["chr", "start", "end"])
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if len(df) == 0 or df.shape[1] < 3:
        return pd.DataFrame(columns=["chr", "start", "end"])
    out = pd.DataFrame({
        "chr": df.iloc[:, 0].astype(str),
        "start": pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).astype(int),
        "end":   pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0).astype(int),
    })
    out = out[(out["start"] >= 0) & (out["end"] > out["start"])]
    return out



def write_attribution_bed(
    raw_path: str, out_bed: str, genes_keep: set, pert: str,
    gene_windows: Dict[str, Tuple[str, int, int]],
) -> int:
    if not os.path.exists(raw_path):
        print(f"[WARN] Missing attribution raw bed: {raw_path}")
        return 0
    raw = pd.read_csv(raw_path, sep=r"\s+", header=None, comment="#", engine="python")
    ncol = int(raw.shape[1])
    if ncol < 3:
        return 0
    df = raw.rename(columns={0: "chr", 1: "start", 2: "end"}).copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["gene"] = df[ncol - 2].astype(str) if ncol >= 2 else ""
    df = df[df["gene"].isin(genes_keep)].copy()
    df = ensure_bed3_safe_df(df)
    if len(df) == 0:
        return 0
    df = filter_peaks_to_tss_window(df, gene_windows)
    if len(df) == 0:
        return 0
    df = df.reset_index(drop=True)
    df["name"] = df["gene"].astype(str) + "|" + str(pert) + "|" + df.index.astype(str)
    safe_makedirs(os.path.dirname(out_bed))
    df[["chr", "start", "end", "name"]].to_csv(out_bed, sep="\t", header=False, index=False)
    return len(df)


def write_re2g_like_bed(re2g_df: pd.DataFrame, genes_keep: set,
                        gene_windows: Dict[str, Tuple[str, int, int]],
                        out_bed: str) -> int:
    sub = re2g_df[re2g_df["gene"].isin(genes_keep)].copy()
    if len(sub) == 0:
        return 0
    sub = sub.drop_duplicates(subset=["chr", "start", "end", "gene"]).copy()
    sub = ensure_bed3_safe_df(sub)
    sub = filter_peaks_to_tss_window(sub, gene_windows)
    if len(sub) == 0:
        return 0
    out = sub[["chr", "start", "end", "name", "score", "strand", "gene"]]
    safe_makedirs(os.path.dirname(out_bed))
    out.to_csv(out_bed, sep="\t", header=False, index=False)
    return len(out)


def write_abc_bed(abc_df: pd.DataFrame, genes_keep: set,
                  gene_windows: Dict[str, Tuple[str, int, int]], out_bed: str) -> int:
    sub = abc_df[abc_df["gene"].isin(genes_keep)].copy()
    if len(sub) == 0:
        return 0
    sub = sub.sort_values("ABC.Score", ascending=False)
    sub = sub.drop_duplicates(subset=["chr", "start", "end", "gene"]).copy()
    out = sub[["chr", "start", "end", "gene", "ABC.Score"]].rename(columns={"ABC.Score": "score"})
    out = ensure_bed3_safe_df(out)
    out = filter_peaks_to_tss_window(out, gene_windows)
    if len(out) == 0:
        return 0
    safe_makedirs(os.path.dirname(out_bed))
    out.to_csv(out_bed, sep="\t", header=False, index=False)
    return len(out)


def write_tss1kbp_bed(base_df: pd.DataFrame, out_bed: str) -> int:
    bed = base_df.copy()
    bed[["start", "end"]] = bed.apply(expand_to_promoter, axis=1)
    bed = filter_primary_chroms_df(bed, chr_col="chr")
    bed = bed[pd.to_numeric(bed["end"], errors="coerce") >
              pd.to_numeric(bed["start"], errors="coerce")]
    safe_makedirs(os.path.dirname(out_bed))
    bed.to_csv(out_bed, sep="\t", header=False, index=False)
    return len(bed)



def _read_peaks_with_gene(path: str, source: str, gene_col: int, score_col: int) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "score", "source"])
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if len(df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "score", "source"])
    out = pd.DataFrame({
        "chr": df.iloc[:, 0].astype(str),
        "start": pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).astype(int),
        "end":   pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0).astype(int),
    })
    out["gene"] = (df.iloc[:, gene_col].astype(str)
                   if gene_col is not None and df.shape[1] > gene_col else "unknown")
    out["score"] = (pd.to_numeric(df.iloc[:, score_col], errors="coerce").fillna(0.0)
                    if score_col is not None and df.shape[1] > score_col else 0.0)
    out["source"] = source
    out = out[(out["start"] >= 0) & (out["end"] > out["start"])]
    return out


def _read_attribution_bed(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "score", "source"])
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if len(df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "score", "source"])
    out = pd.DataFrame({
        "chr": df.iloc[:, 0].astype(str),
        "start": pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).astype(int),
        "end":   pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0).astype(int),
    })
    out["gene"] = (df.iloc[:, 3].astype(str).str.split("|", n=1, expand=True)[0]
                   if df.shape[1] >= 4 else "unknown")
    out["score"] = 0.0
    out["source"] = "Attribution"
    out = out[(out["start"] >= 0) & (out["end"] > out["start"])]
    return out


def merge_peaks(peaks_list: list, genes_keep: set) -> pd.DataFrame:
    if not peaks_list:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "sources", "max_score"])
    all_peaks = pd.concat(peaks_list, ignore_index=True)
    if len(all_peaks) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "sources", "max_score"])
    if genes_keep:
        all_peaks = all_peaks[all_peaks["gene"].isin(genes_keep)].copy()
    if len(all_peaks) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "sources", "max_score"])
    return merge_intervals_by_gene(
        all_peaks[["chr", "start", "end", "gene", "score", "source"]]
    )


def label_atac_overlap(peaks_df: pd.DataFrame, atac_df: pd.DataFrame) -> pd.DataFrame:
    if len(peaks_df) == 0:
        peaks_df = peaks_df.copy()
        peaks_df["atac_overlap"] = pd.Series(dtype=int)
        return peaks_df
    if len(atac_df) == 0:
        peaks_df = peaks_df.copy()
        peaks_df["atac_overlap"] = 0
        return peaks_df
    peaks_bt = pybedtools.BedTool.from_dataframe(peaks_df[["chr", "start", "end"]])
    atac_bt = pybedtools.BedTool.from_dataframe(atac_df[["chr", "start", "end"]])
    overlap_bt = peaks_bt.intersect(atac_bt, u=True, wa=True)
    if len(overlap_bt) > 0:
        odf = overlap_bt.to_dataframe(names=["chr", "start", "end"])
        overlap_set = set(zip(odf["chr"].astype(str), odf["start"], odf["end"]))
    else:
        overlap_set = set()
    peaks_df = peaks_df.copy()
    peaks_df["atac_overlap"] = peaks_df.apply(
        lambda r: 1 if (str(r["chr"]), int(r["start"]), int(r["end"])) in overlap_set else 0,
        axis=1,
    )
    return peaks_df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_name", default="MartinRufino2025_mixscape_exnp_train")
    ap.add_argument("--study_suffix", default="alphagenome_transfer_epoch100_batch256_adamw5e3")
    ap.add_argument("--pretrained_model", default="alphagenome",
                    choices=["alphagenome", "borzoi", "enformer"])
    ap.add_argument("--re2g_data_path", default="data/ENCFF497HEA.bed.gz")
    ap.add_argument("--re2g_extended_data_path", default="data/ENCFF269DKY.bed.gz")
    ap.add_argument("--abc_data_path",
                    default="data/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.hg38_replaced.txt.gz")
    ap.add_argument("--abc_cell_type", default="K562-Roadmap")
    ap.add_argument("--gtgenes_xlsx",
                    default="data/science.ads7951_tables_s1_to_s6/science.ads7951_table_s3.xlsx")
    ap.add_argument("--atac_root", default="reference/martin_atac")
    ap.add_argument("--attribution_root", default="attribution")
    ap.add_argument("--output_root", default="cre_gtgenes")
    ap.add_argument("--attribution_suffix", default="",
                    help="Read {pert}{suffix}_peaks.bed instead of {pert}_peaks.bed (e.g. '_gtgenes')")
    ap.add_argument("--tfs", default="")
    args = ap.parse_args()

    study = f"{args.study_name}__{args.study_suffix}"
    out_study_dir = os.path.join(args.output_root, study)
    unified_dir = os.path.join(out_study_dir, "unified_peaks")
    safe_makedirs(unified_dir)

    base_bed_path = f"fasta/{args.study_name}.bed"
    TSS_FLANK = MODEL_CONTEXT_LENGTH[args.pretrained_model] // 2
    chrom_sizes = load_chrom_sizes("fasta/GRCh38.p14.genome.fa.sizes")

    if args.tfs.strip():
        perts = [x.strip() for x in args.tfs.split(",") if x.strip()]
    else:
        attr_root = os.path.join(args.attribution_root, study)
        if not os.path.isdir(attr_root):
            raise FileNotFoundError(f"[ERROR] Missing directory: {attr_root}")
        perts = sorted([p for p in os.listdir(attr_root)
                        if os.path.isdir(os.path.join(attr_root, p))])
    print(f"[INFO] Found {len(perts)} perturbations")

    print(f"[INFO] Loading rE2G from {args.re2g_data_path}")
    re2g_df = load_re2g_data(args.re2g_data_path)
    print(f"[INFO] rE2G rows: {len(re2g_df)}")

    print(f"[INFO] Loading rE2G extended from {args.re2g_extended_data_path}")
    re2g_ext_df = load_re2g_data(args.re2g_extended_data_path)
    print(f"[INFO] rE2G extended rows: {len(re2g_ext_df)}")

    abc_df = load_abc_data(args.abc_data_path, args.abc_cell_type)

    print(f"[INFO] Loading gtgenes from {args.gtgenes_xlsx}")
    gtgenes_by_pert = load_table_s3_genes(args.gtgenes_xlsx)
    print(f"[INFO] Table_S3 perturbations: {len(gtgenes_by_pert)}")

    base_full = pd.read_csv(
        base_bed_path, sep="\t", header=None,
        names=["chr", "start", "end", "gene", "score", "strand", "split"],
    )

    summary_rows = []
    for pert in perts:
        tf_symbol = pert.split(".")[-1]
        print(f"\n[INFO] === {pert} (TF={tf_symbol}) ===")

        if tf_symbol not in gtgenes_by_pert:
            print(f"[WARN] {tf_symbol} not in Table_S3. Skipping.")
            continue
        gtgenes_set = gtgenes_by_pert[tf_symbol]
        gtgenes = sorted(gtgenes_set)
        print(f"  gtgenes from Table_S3: {len(gtgenes)}")

        atac_bed_path = os.path.join(args.atac_root, f"{tf_symbol}.bed")
        atac_df = load_atac_bed(atac_bed_path)
        print(f"  ATAC ground-truth: {len(atac_df)} peaks ({atac_bed_path})")
        if len(atac_df) == 0:
            print(f"[WARN] Missing/empty ATAC BED for {pert}. Skipping.")
            continue

        base_df = base_full[base_full["gene"].isin(gtgenes_set)].copy()
        base_df = filter_primary_chroms_df(base_df, chr_col="chr")
        if len(base_df) == 0:
            print(f"[WARN] No gtgenes found in base bed for {pert}. Skipping.")
            continue
        base_df[["win_start", "win_end"]] = base_df.apply(
            lambda r: expand_to_tss_window(r, flank=TSS_FLANK), axis=1
        )
        win_df = base_df[["chr", "win_start", "win_end", "gene"]].rename(
            columns={"win_start": "start", "win_end": "end"}
        )
        win_df = ensure_bed3_safe_df(win_df)
        gene_windows = {
            r["gene"]: (r["chr"], int(r["start"]), int(r["end"]))
            for _, r in win_df.iterrows()
        }
        gene_windows = clip_gene_windows(gene_windows, chrom_sizes)
        n_win = len(gene_windows)
        print(f"  gtgenes with TSS in base bed: {n_win}")

        outdir = os.path.join(out_study_dir, pert)
        safe_makedirs(outdir)

        attr_raw = os.path.join(args.attribution_root, study, pert,
                                 f"{pert}{args.attribution_suffix}_peaks.bed")
        attr_bed = os.path.join(outdir, f"attribution_{pert}.bed")
        n_attr = write_attribution_bed(attr_raw, attr_bed, gtgenes_set, pert, gene_windows)
        print(f"  attribution: {n_attr} regions")

        re2g_bed = os.path.join(outdir, f"re2g_{pert}.bed")
        n_re2g = write_re2g_like_bed(re2g_df, gtgenes_set, gene_windows, re2g_bed)
        print(f"  rE2G: {n_re2g} regions")

        re2g_ext_bed = os.path.join(outdir, f"re2g_extended_{pert}.bed")
        n_re2g_ext = write_re2g_like_bed(re2g_ext_df, gtgenes_set, gene_windows, re2g_ext_bed)
        print(f"  rE2G extended: {n_re2g_ext} regions")

        abc_bed = os.path.join(outdir, f"abc_score_{pert}.bed")
        n_abc = write_abc_bed(abc_df, gtgenes_set, gene_windows, abc_bed)
        print(f"  ABC: {n_abc} regions")

        tss_bed = os.path.join(outdir, f"tss_1kbp_{pert}.bed")
        n_tss = write_tss1kbp_bed(base_df.drop(columns=["win_start", "win_end"], errors="ignore"),
                                   tss_bed)
        print(f"  TSS±1kbp: {n_tss} regions")

        peaks_list = []
        attr_peaks = _read_attribution_bed(attr_bed)
        if len(attr_peaks) > 0:
            peaks_list.append(attr_peaks[attr_peaks["gene"].isin(gtgenes_set)])
        re2g_peaks = _read_peaks_with_gene(re2g_bed, "rE2G", gene_col=6, score_col=4)
        if len(re2g_peaks) > 0:
            peaks_list.append(re2g_peaks[re2g_peaks["gene"].isin(gtgenes_set)])
        re2g_ext_peaks = _read_peaks_with_gene(re2g_ext_bed, "rE2G_extended",
                                                gene_col=6, score_col=4)
        if len(re2g_ext_peaks) > 0:
            peaks_list.append(re2g_ext_peaks[re2g_ext_peaks["gene"].isin(gtgenes_set)])
        abc_peaks = _read_peaks_with_gene(abc_bed, "ABC", gene_col=3, score_col=4)
        if len(abc_peaks) > 0:
            peaks_list.append(abc_peaks[abc_peaks["gene"].isin(gtgenes_set)])

        unified = merge_peaks(peaks_list, gtgenes_set)
        print(f"  unified peaks: {len(unified)}")
        if len(unified) == 0:
            print(f"[WARN] No unified peaks for {pert}. Skipping.")
            continue

        unified = label_atac_overlap(unified, atac_df)
        n_pos = int(unified["atac_overlap"].sum())
        print(f"  ATAC overlap (positive): {n_pos}")

        out_bed = os.path.join(unified_dir, f"unified_peaks_{pert}.bed")
        unified.to_csv(out_bed, sep="\t", header=False, index=False)
        print(f"  Saved: {out_bed}")

        srow = {
            "study": study, "pert": pert, "tf_symbol": tf_symbol,
            "n_gtgenes": len(gtgenes), "n_gtgenes_with_tss": n_win,
            "n_attribution": n_attr, "n_re2g": n_re2g,
            "n_re2g_extended": n_re2g_ext, "n_abc": n_abc,
            "n_atac_rows": len(atac_df),
            "n_unified_peaks": len(unified), "n_positive": n_pos,
        }
        summary_rows.append(srow)
        with open(os.path.join(unified_dir, f"summary_{pert}.json"), "w") as fh:
            json.dump(srow, fh, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = os.path.join(unified_dir, "unified_peaks_summary.tsv")
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    print(f"\n[INFO] Saved overall summary: {summary_tsv}")
    print(f"[INFO] Processed {len(summary_rows)} TFs")


if __name__ == "__main__":
    main()
