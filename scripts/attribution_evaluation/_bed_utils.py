#!/usr/bin/env python
import os
from typing import Tuple, Set, Dict

import pandas as pd
import numpy as np
import pybedtools

ALLOW_CHROMS = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])

MODEL_CONTEXT_LENGTH = {
    "alphagenome": 1_048_576,
    "borzoi": 524_288,
    "enformer": 196_608,
}

BED_TRACK_COLORS = {
    'attribution_re2g_extended': '#4A0080',
    'attribution_re2g_extended_shuffle': '#D8B4E8',
    'attribution_re2g': '#660099',
    'attribution_re2g_shuffle': '#E6CCF5',
    'attribution_abc': '#8B0000',
    'attribution_abc_shuffle': '#FFB6C1',
    'attribution': '#E74C3C',
    'attribution_shuffle': '#FADBD8',
    're2g_extended': '#00695C',
    're2g_extended_shuffle': '#B2DFDB',
    're2g': '#00796B',
    're2g_shuffle': '#B2DFDB',
    'tss_1kbp': '#3498DB',
    'abc_score': '#2ECC71',
    'abc_score_shuffle': '#D5F5E3',
    'fanta_bio': '#9B59B6',
    'fanta_bio_shuffle': '#E8DAEF',
    'chip_seq': '#F39C12',
}



def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)



def load_chrom_sizes(sizes_path: str) -> Dict[str, int]:
    chrom_sizes = {}
    with open(sizes_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def clip_gene_windows(
    gene_windows: Dict[str, Tuple[str, int, int]],
    chrom_sizes: Dict[str, int],
) -> Dict[str, Tuple[str, int, int]]:
    clipped = {}
    for gene, (chrom, start, end) in gene_windows.items():
        if chrom in chrom_sizes:
            end = min(end, chrom_sizes[chrom])
        start = max(0, start)
        if end > start:
            clipped[gene] = (chrom, start, end)
    return clipped



def filter_primary_chroms_df(df: pd.DataFrame, chr_col: str = "chr") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if chr_col not in df.columns:
        return df
    return df[df[chr_col].astype(str).isin(ALLOW_CHROMS)].copy()


def filter_primary_chroms_bt(bt: pybedtools.BedTool) -> pybedtools.BedTool:
    return bt.filter(lambda x: x.chrom in ALLOW_CHROMS).saveas()



def ensure_bed3_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = filter_primary_chroms_df(out, chr_col="chr")
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(int)
    out["end"]   = pd.to_numeric(out["end"], errors="coerce").fillna(0).astype(int)
    out["start"] = out["start"].where(out["start"] >= 0, 0)
    out = out[out["end"] > out["start"]].copy()
    return out


def merge_bed_regions(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = ensure_bed3_safe_df(df)
    if len(df) == 0:
        return df
    bt = pybedtools.BedTool.from_dataframe(df[["chr", "start", "end"]]).sort().merge()
    if len(bt) == 0:
        return pd.DataFrame(columns=["chr", "start", "end"])
    merged_df = bt.to_dataframe()
    merged_df.columns = ["chr", "start", "end"]
    return merged_df



def load_chip_union_for_tf(
    chipatlas_dir: str, peak_list_tsv: str, tf: str, pval_suffix: str = ".05"
) -> pybedtools.BedTool:
    peak_list = pd.read_csv(peak_list_tsv, sep="\t")
    srxs = peak_list.query("Target == @tf")["SRX"].dropna().unique().tolist()

    beds = []
    for srx in srxs:
        path = os.path.join(chipatlas_dir, f"{srx}{pval_suffix}.bed")
        if not os.path.exists(path):
            path_ex = os.path.join(chipatlas_dir, f"{srx}.ex{pval_suffix}.bed")
            if os.path.exists(path_ex):
                path = path_ex
            else:
                print(f"[WARN] Missing ChIP-seq bed: {path} (and .ex). Skipping SRX={srx}")
                continue

        chip_df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
        chip_df.columns = ["chr", "start", "end"]
        chip_df = filter_primary_chroms_df(chip_df, chr_col="chr")
        chip_df = ensure_bed3_safe_df(chip_df)

        if len(chip_df) == 0:
            continue
        beds.append(pybedtools.BedTool.from_dataframe(chip_df[["chr", "start", "end"]]))

    if len(beds) == 0:
        raise FileNotFoundError(f"[ERROR] No ChIP-seq peaks found for TF={tf}")

    u = beds[0]
    for b in beds[1:]:
        u = u.cat(b, postmerge=False)

    return filter_primary_chroms_bt(u.sort().merge())



def expand_to_tss_window(row: pd.Series, flank: int) -> pd.Series:
    if row["strand"] == "+":
        tss = int(row["start"])
    else:
        tss = int(row["end"])
    new_start = max(0, tss - flank)
    new_end = tss + flank
    return pd.Series([new_start, new_end])


def expand_to_promoter(row: pd.Series, flank: int = 1000) -> pd.Series:
    if row["strand"] == "+":
        tss = int(row["start"])
    else:
        tss = int(row["end"])
    new_start = max(0, tss - flank)
    new_end = tss + flank
    return pd.Series([new_start, new_end])



def shuffled_path_for(bed_path: str) -> str:
    if bed_path.endswith(".bed"):
        return bed_path[:-4] + ".shuffle.bed"
    return bed_path + ".shuffle.bed"


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return (a_start < b_end) and (b_start < a_end)


def filter_peaks_to_tss_window(
    peaks_df: pd.DataFrame,
    gene_windows: Dict[str, Tuple[str, int, int]],
) -> pd.DataFrame:
    if len(peaks_df) == 0:
        return peaks_df

    out_rows = []
    for gene, gdf in peaks_df.groupby("gene"):
        if gene not in gene_windows:
            continue
        w_chr, w_start, w_end = gene_windows[gene]
        gdf = gdf.copy()
        gdf = gdf[gdf["chr"] == w_chr].copy()
        gdf = gdf[
            (gdf["end"].astype(int) > w_start) & (gdf["start"].astype(int) < w_end)
        ].copy()
        if len(gdf) > 0:
            out_rows.append(gdf)

    if len(out_rows) == 0:
        return pd.DataFrame(columns=peaks_df.columns)
    return pd.concat(out_rows, ignore_index=True)


def gene_wise_shuffle_peaks(
    peaks_df: pd.DataFrame,
    gene_windows: Dict[str, Tuple[str, int, int]],
    seed: int,
    no_overlap: bool = True,
    max_tries_per_peak: int = 2000,
    filter_to_window: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_rows = []

    for gene, gdf in peaks_df.groupby("gene"):
        if gene not in gene_windows:
            continue

        w_chr, w_start, w_end = gene_windows[gene]
        gdf = gdf.copy()

        if filter_to_window:
            gdf = gdf[gdf["chr"] == w_chr].copy()
            gdf = gdf[
                (gdf["end"].astype(int) > w_start) & (gdf["start"].astype(int) < w_end)
            ].copy()
            if len(gdf) == 0:
                continue

        lens = (gdf["end"].astype(int) - gdf["start"].astype(int)).to_numpy()
        max_len = int(w_end - w_start)
        ok_mask = lens <= max_len
        if not np.all(ok_mask):
            n_drop = int((~ok_mask).sum())
            print(f"[WARN] Gene-wise shuffle: dropping {n_drop} peaks (too long for window) for gene={gene}")
            gdf = gdf.loc[ok_mask].copy()
            lens = lens[ok_mask]
        if len(gdf) == 0:
            continue

        placed = []
        for i, L in enumerate(lens):
            L = int(L)
            lo = int(w_start)
            hi = int(w_end - L)
            if hi <= lo:
                print(f"[WARN] Gene-wise shuffle: cannot place peak length={L} in window for gene={gene}")
                continue

            placed_start = None
            placed_end = None

            if no_overlap:
                for _ in range(max_tries_per_peak):
                    s = int(rng.integers(lo, hi + 1))
                    e = s + L
                    if all(not _intervals_overlap(s, e, ps, pe) for ps, pe in placed):
                        placed_start, placed_end = s, e
                        placed.append((placed_start, placed_end))
                        break

            if placed_start is None:
                if no_overlap:
                    print(f"[WARN] Gene-wise shuffle: failed no-overlap placement for gene={gene}. Falling back to allow-overlap.")
                s = int(rng.integers(lo, hi + 1))
                e = s + L
                placed_start, placed_end = s, e
                placed.append((placed_start, placed_end))

            row = gdf.iloc[i].copy()
            row["chr"] = w_chr
            row["start"] = placed_start
            row["end"] = placed_end
            out_rows.append(row)

    if len(out_rows) == 0:
        return pd.DataFrame(columns=peaks_df.columns)

    out_df = pd.DataFrame(out_rows)
    out_df = ensure_bed3_safe_df(out_df)
    return out_df



def load_attribution_peaks_with_gene(
    tf_peaks_bed: str, top_genes: Set[str]
) -> Tuple[pd.DataFrame, int]:
    if not os.path.exists(tf_peaks_bed):
        return pd.DataFrame(), 0

    raw = pd.read_csv(tf_peaks_bed, sep=r"\s+", header=None, comment="#", engine="python")
    ncol = int(raw.shape[1])
    if ncol < 5:
        print(f"[WARN] Attribution peaks bed has too few columns: {tf_peaks_bed} (cols={ncol})")
        return pd.DataFrame(), ncol

    df = raw.copy()
    df = df.rename(columns={0: "chr", 1: "start", 2: "end"})
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)

    gene_col = ncol - 2
    df["gene"] = df[gene_col].astype(str)

    df = df[df["gene"].isin(top_genes)].copy()
    df = ensure_bed3_safe_df(df)
    return df, ncol



def get_bed_track_configs(study: str, pert: str, tf_symbol: str) -> list:
    return [
        {
            'path': f"cre/{study}/{pert}/attribution_re2g_extended_{pert}.bed",
            'name': 'attribution_re2g_extended',
            'color': BED_TRACK_COLORS['attribution_re2g_extended'],
            'label': 'Attribution + rE2G extended',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_re2g_extended_{pert}.shuffle.bed",
            'name': 'attribution_re2g_extended_shuffle',
            'color': BED_TRACK_COLORS['attribution_re2g_extended_shuffle'],
            'label': 'Attribution + rE2G extended (shuffle)',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_re2g_{pert}.bed",
            'name': 'attribution_re2g',
            'color': BED_TRACK_COLORS['attribution_re2g'],
            'label': 'Attribution + rE2G',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_re2g_{pert}.shuffle.bed",
            'name': 'attribution_re2g_shuffle',
            'color': BED_TRACK_COLORS['attribution_re2g_shuffle'],
            'label': 'Attribution + rE2G (shuffle)',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_abc_{pert}.bed",
            'name': 'attribution_abc',
            'color': BED_TRACK_COLORS['attribution_abc'],
            'label': 'Attribution + ABC score',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_abc_{pert}.shuffle.bed",
            'name': 'attribution_abc_shuffle',
            'color': BED_TRACK_COLORS['attribution_abc_shuffle'],
            'label': 'Attribution + ABC score (shuffle)',
            'gene_col': 4,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_{pert}.bed",
            'name': 'attribution',
            'color': BED_TRACK_COLORS['attribution'],
            'label': 'Attribution',
            'gene_col': None,
        },
        {
            'path': f"cre/{study}/{pert}/attribution_{pert}.shuffle.bed",
            'name': 'attribution_shuffle',
            'color': BED_TRACK_COLORS['attribution_shuffle'],
            'label': 'Attribution (shuffle)',
            'gene_col': None,
        },
        {
            'path': f"cre/{study}/{pert}/re2g_extended_{pert}.bed",
            'name': 're2g_extended',
            'color': BED_TRACK_COLORS['re2g_extended'],
            'label': 'rE2G extended',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/re2g_extended_{pert}.shuffle.bed",
            'name': 're2g_extended_shuffle',
            'color': BED_TRACK_COLORS['re2g_extended_shuffle'],
            'label': 'rE2G extended (shuffle)',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/re2g_{pert}.bed",
            'name': 're2g',
            'color': BED_TRACK_COLORS['re2g'],
            'label': 'rE2G',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/re2g_{pert}.shuffle.bed",
            'name': 're2g_shuffle',
            'color': BED_TRACK_COLORS['re2g_shuffle'],
            'label': 'rE2G (shuffle)',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/abc_score_{pert}.bed",
            'name': 'abc_score',
            'color': BED_TRACK_COLORS['abc_score'],
            'label': 'ABC score',
            'gene_col': 3,
        },
        {
            'path': f"cre/{study}/{pert}/abc_score_{pert}.shuffle.bed",
            'name': 'abc_score_shuffle',
            'color': BED_TRACK_COLORS['abc_score_shuffle'],
            'label': 'ABC score (shuffle)',
            'gene_col': 3,
        },
        {
            'path': f"cre/{study}/{pert}/fanta_bio_{pert}.bed",
            'name': 'fanta_bio',
            'color': BED_TRACK_COLORS['fanta_bio'],
            'label': 'fanta.bio',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/fanta_bio_{pert}.shuffle.bed",
            'name': 'fanta_bio_shuffle',
            'color': BED_TRACK_COLORS['fanta_bio_shuffle'],
            'label': 'fanta.bio (shuffle)',
            'gene_col': 6,
        },
        {
            'path': f"cre/{study}/{pert}/tss_1kbp_{pert}.bed",
            'name': 'tss_1kbp',
            'color': BED_TRACK_COLORS['tss_1kbp'],
            'label': 'TSS±1kbp',
            'gene_col': 3,
        },
        {
            'path': f"cre/{study}/{pert}/chip_seq_{tf_symbol}.bed",
            'name': 'chip_seq',
            'color': BED_TRACK_COLORS['chip_seq'],
            'label': f'ChIP-seq ({tf_symbol})',
            'gene_col': None,
        },
    ]
