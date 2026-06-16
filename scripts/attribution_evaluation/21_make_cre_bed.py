#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Set, Dict

import pandas as pd
import numpy as np
import pybedtools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bed_utils import (  # noqa: E402
    MODEL_CONTEXT_LENGTH,
    load_chrom_sizes,
    clip_gene_windows,
    filter_primary_chroms_df,
    filter_primary_chroms_bt,
    ensure_bed3_safe_df,
    load_chip_union_for_tf,
    expand_to_tss_window,
    expand_to_promoter,
    shuffled_path_for,
    filter_peaks_to_tss_window,
    gene_wise_shuffle_peaks,
    load_attribution_peaks_with_gene,
)


GENEWISE_SHUFFLE_MAX_TRIES_PER_PEAK = 2000
GENEWISE_SHUFFLE_NO_OVERLAP = False
GENEWISE_SHUFFLE_SEED_BASE = 12345

_STANDARD_BED_FORMATS = {
    "abc_score": {
        "min_cols": 5,
        "cols": ["chr", "start", "end", "gene", "score"],
    },
    "fanta_bio": {
        "min_cols": 7,
        "cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
    },
    "re2g_extended": {
        "min_cols": 7,
        "cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
    },
    "re2g": {
        "min_cols": 7,
        "cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
    },
}



def parse_and_write_attribution_bed_for_gimme(
    tf_peaks_bed: str,
    out_bed: str,
    top_genes: Set[str],
    pert: str,
    gene_windows: Dict[str, Tuple[str, int, int]] = None,
) -> Tuple[pd.DataFrame, int]:
    if not os.path.exists(tf_peaks_bed):
        return pd.DataFrame(), 0

    raw = pd.read_csv(tf_peaks_bed, sep=r"\s+", header=None, comment="#", engine="python")
    ncol = int(raw.shape[1])
    if ncol < 3:
        print(f"[WARN] Attribution bed has <3 columns: {tf_peaks_bed} (cols={ncol})")
        return pd.DataFrame(), ncol

    df = raw.copy()
    df = df.rename(columns={0: "chr", 1: "start", 2: "end"})
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)

    if ncol >= 2:
        gene_col = ncol - 2
        df["gene"] = df[gene_col].astype(str)
    else:
        df["gene"] = ""

    df = df[df["gene"].isin(top_genes)].copy()
    df = ensure_bed3_safe_df(df)

    if len(df) == 0:
        print(f"[WARN] Attribution: no peaks left after parsing+gene-filter: {tf_peaks_bed}")
        return df, ncol

    if gene_windows is not None:
        df = filter_peaks_to_tss_window(df, gene_windows)
        if len(df) == 0:
            print(f"[WARN] Attribution: no peaks left after TSS window filter: {tf_peaks_bed}")
            return df, ncol

    df = df.reset_index(drop=True)
    df["name"] = df["gene"].astype(str) + "|" + str(pert) + "|" + df.index.astype(str)

    os.makedirs(os.path.dirname(out_bed), exist_ok=True)
    df[["chr", "start", "end", "name"]].to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote Attribution bed for gimme: {out_bed} ({len(df)} regions)")

    return df, ncol


def load_re2g_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, sep="\t", comment="#", header=None, compression="gzip")
    df = df.rename(columns={0: "chr", 1: "start", 2: "end", 3: "name", 4: "class", 5: "TargetGene"})
    df["score"] = df.iloc[:, -1]
    df["strand"] = "."
    df = filter_primary_chroms_df(df, chr_col="chr")
    df["gene"] = df["TargetGene"].astype(str)
    df["gene"] = df["gene"].replace("NA", pd.NA)
    df_with_genes = df.dropna(subset=["gene"]).copy()
    return df, df_with_genes



def _create_gene_filtered_bed(
    source_df: pd.DataFrame,
    genes: list,
    gene_windows: dict,
    out_cols: list,
    bed_path: str,
    label: str,
) -> bool:
    filtered = source_df[source_df["gene"].isin(genes)].copy()
    if len(filtered) == 0:
        print(f"[WARN] No {label} rows. Skipping.")
        return False

    unique = filtered.drop_duplicates(subset=["chr", "start", "end", "gene"]).copy()
    bed_output = unique[out_cols].copy()
    bed_output = ensure_bed3_safe_df(bed_output)
    bed_output = filter_peaks_to_tss_window(bed_output, gene_windows)
    if len(bed_output) == 0:
        print(f"[WARN] No {label} regions in TSS windows. Skipping.")
        return False

    bed_output.to_csv(bed_path, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote: {bed_path} ({len(bed_output)} regions; original {len(filtered)})")
    return True


def _load_bed_as_merge_source(bed_path: str, source_type: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["chr", "start", "end", "name", "gene"])
    if not os.path.exists(bed_path):
        return empty

    df = pd.read_csv(bed_path, sep="\t", header=None)

    if source_type == "attribution":
        if df.shape[1] < 4:
            return empty
        df = df.iloc[:, :4].copy()
        df.columns = ["chr", "start", "end", "name"]
        df["gene"] = df["name"].astype(str).str.split("|", n=1).str[0]
    elif source_type in ("re2g", "re2g_extended", "fanta_bio"):
        if df.shape[1] < 7:
            return empty
        df = df.iloc[:, :7].copy()
        df.columns = ["chr", "start", "end", "name", "score", "strand", "gene"]
        label = {"re2g_extended": "rE2G_ext", "re2g": "rE2G", "fanta_bio": "fanta_bio"}[source_type]
        df["name"] = df["gene"].astype(str) + f"|{label}|" + df.index.astype(str)
    elif source_type == "abc_score":
        if df.shape[1] < 5:
            return empty
        df = df.iloc[:, :5].copy()
        df.columns = ["chr", "start", "end", "gene", "score"]
        df["name"] = df["gene"].astype(str) + "|ABC|" + df.index.astype(str)
    else:
        return empty

    df = ensure_bed3_safe_df(df)
    return df[["chr", "start", "end", "name", "gene"]]


def _create_merged_bed(
    source_a_path: str, source_a_type: str,
    source_b_path: str, source_b_type: str,
    bed_path: str, label: str,
) -> bool:
    if not os.path.exists(source_a_path):
        print(f"[WARN] Source bed not found: {source_a_path} (skip {label})")
        return False
    if not os.path.exists(source_b_path):
        print(f"[WARN] Source bed not found: {source_b_path} (skip {label})")
        return False

    print(f"[INFO] Creating {label} merged bed")
    a_df = _load_bed_as_merge_source(source_a_path, source_a_type)
    b_df = _load_bed_as_merge_source(source_b_path, source_b_type)

    dfs = [d for d in [a_df, b_df] if len(d) > 0]
    if len(dfs) == 0:
        print(f"[WARN] No regions to merge for {label}: {bed_path}")
        return False

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["chr", "start", "end", "gene"])
    merged = ensure_bed3_safe_df(merged)
    merged = merged.sort_values(["chr", "start", "end"])
    merged[["chr", "start", "end", "name", "gene"]].to_csv(bed_path, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote: {bed_path} ({len(merged)} regions; a={len(a_df)}, b={len(b_df)})")
    return True



def _shuffle_standard_bed(
    bed_name: str, bed_path: str, gene_windows: dict, seed: int,
) -> None:
    fmt = _STANDARD_BED_FORMATS.get(bed_name)
    if fmt is None:
        print(f"[WARN] No format defined for standard shuffle: {bed_name}")
        return

    shuffle_path = shuffled_path_for(bed_path)
    if not os.path.exists(bed_path):
        print(f"[WARN] Missing bed for shuffle: {bed_path}")
        return

    b = pd.read_csv(bed_path, sep="\t", header=None)
    min_cols = fmt["min_cols"]
    cols = fmt["cols"]
    if b.shape[1] < min_cols:
        print(f"[WARN] {bed_name} bed has <{min_cols} cols: {bed_path} (skip shuffle)")
        return

    b = b.iloc[:, :len(cols)].copy()
    b.columns = cols
    b = ensure_bed3_safe_df(b)
    b = b[b["gene"].isin(gene_windows.keys())].copy()
    if len(b) == 0:
        print(f"[WARN] No peaks left after gene filter for shuffle: {bed_path}")
        return

    shuf = gene_wise_shuffle_peaks(
        peaks_df=b,
        gene_windows=gene_windows,
        seed=seed,
        no_overlap=GENEWISE_SHUFFLE_NO_OVERLAP,
        max_tries_per_peak=GENEWISE_SHUFFLE_MAX_TRIES_PER_PEAK,
        filter_to_window=True,
    )
    if len(shuf) == 0:
        print(f"[WARN] Gene-wise shuffle produced empty result: {shuffle_path}")
        return

    shuf = ensure_bed3_safe_df(shuf)
    shuf[cols].to_csv(shuffle_path, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote gene-wise shuffle: {shuffle_path} ({len(shuf)} regions)")


def _shuffle_attribution_bed(
    cfg: dict, top_genes: set, gene_windows: dict, seed: int, pert: str,
) -> None:
    shuffle_path = shuffled_path_for(cfg["bed_path"])
    raw_path = cfg["bed_path_raw"]
    peaks_df, ncol = load_attribution_peaks_with_gene(raw_path, top_genes=top_genes)
    if len(peaks_df) == 0:
        print(f"[WARN] Attribution peaks empty after gene filter (or missing): {raw_path} (skip shuffle)")
        return

    peaks_df = peaks_df.copy().reset_index(drop=True)
    peaks_df["name"] = peaks_df["gene"].astype(str) + "|" + str(pert) + "|" + peaks_df.index.astype(str)

    shuf = gene_wise_shuffle_peaks(
        peaks_df=peaks_df,
        gene_windows=gene_windows,
        seed=seed,
        no_overlap=GENEWISE_SHUFFLE_NO_OVERLAP,
        max_tries_per_peak=GENEWISE_SHUFFLE_MAX_TRIES_PER_PEAK,
        filter_to_window=True,
    )
    if len(shuf) == 0:
        print(f"[WARN] Gene-wise shuffle produced empty result: {shuffle_path}")
        return

    out_bed = shuf.sort_values(["chr", "start", "end"])[["chr", "start", "end", "name"]].copy()
    out_bed = ensure_bed3_safe_df(out_bed)
    out_bed.to_csv(shuffle_path, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote gene-wise shuffle (BED4): {shuffle_path} ({len(out_bed)} regions)")


def _shuffle_merged_bed(
    cfg: dict,
    source_key_a: str, source_type_a: str,
    source_key_b: str, source_type_b: str,
) -> None:
    shuffle_path = shuffled_path_for(cfg["bed_path"])
    source_a = cfg.get(source_key_a)
    source_b = cfg.get(source_key_b)
    a_shuf = shuffled_path_for(source_a) if source_a else None
    b_shuf = shuffled_path_for(source_b) if source_b else None

    dfs_to_merge = []
    if a_shuf:
        a_df = _load_bed_as_merge_source(a_shuf, source_type_a)
        if len(a_df) > 0:
            dfs_to_merge.append(a_df)
    if b_shuf:
        b_df = _load_bed_as_merge_source(b_shuf, source_type_b)
        if len(b_df) > 0:
            dfs_to_merge.append(b_df)

    if len(dfs_to_merge) == 0:
        print(f"[WARN] No shuffle source beds found for {cfg['name']}: {shuffle_path}")
        return

    merged = pd.concat(dfs_to_merge, ignore_index=True)
    merged = merged.drop_duplicates(subset=["chr", "start", "end", "gene"])
    merged = ensure_bed3_safe_df(merged)
    merged = merged.sort_values(["chr", "start", "end"])
    merged[["chr", "start", "end", "name", "gene"]].to_csv(shuffle_path, sep="\t", header=False, index=False)
    print(f"[INFO] Wrote {cfg['name']} shuffle: {shuffle_path} ({len(merged)} regions)")



def main():
    ap = argparse.ArgumentParser(description="Create CRE BED files for attribution evaluation")
    ap.add_argument("--study_name", default="NormanWeissman2019_filtered_mixscape_exnp_train")
    ap.add_argument("--pretrained_model", default="alphagenome",
                    choices=["alphagenome", "borzoi", "enformer"])
    ap.add_argument("--study_suffix", default="",
                    help="Study suffix. Default: {model}_transfer_epoch100_batch256_adamw5e3")
    ap.add_argument("--abc_data_path",
                    default="data/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.hg38_replaced.txt.gz")
    ap.add_argument("--abc_cell_type", default="K562-Roadmap")
    ap.add_argument("--cre_data_path", default="data/human-CREv1.1.0.hg38.cre-peaks.with_genes.bed")
    ap.add_argument("--re2g_extended_data_path", default="data/ENCFF269DKY.bed.gz")
    ap.add_argument("--re2g_data_path", default="data/ENCFF497HEA.bed.gz")
    ap.add_argument("--chipatlas_dir", default="reference/chipatlas")
    ap.add_argument("--chipatlas_list", default="reference/chipatlas/download_data_list.tsv")
    ap.add_argument("--chip_pval_suffix", default=".05")
    args = ap.parse_args()

    study_name = args.study_name
    pretrained_model = args.pretrained_model
    study_suffix = args.study_suffix or f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
    study = f"{study_name}__{study_suffix}"
    base_bed = f"fasta/{study_name}.bed"
    TSS_WINDOW_FLANK = MODEL_CONTEXT_LENGTH[pretrained_model] // 2
    chrom_sizes = load_chrom_sizes("fasta/GRCh38.p14.genome.fa.sizes")

    df  = pd.read_csv(f"data/{study_name}.tsv", sep="\t", index_col=0)
    pred = np.load(f"prediction/{study}/prediction.npy")

    df2 = pd.DataFrame(pred, index=df.index)
    df2.columns = df.columns[1:]

    df_val = df[df.columns[1:]]
    ctrl = df_val.columns[0]
    df3 = (df_val.T - df_val[ctrl]).T.drop(ctrl, axis=1)

    tf_list = pd.read_csv(
        "reference/humantfs/DatabaseExtract_v_1.01.txt",
        sep="\t", usecols=["HGNC symbol"]
    )["HGNC symbol"].to_list()
    tfs = [i for i in os.listdir(f"attribution/{study}/") if any(j in i for j in tf_list)]

    print("[INFO] Loading ABC score data...")
    abc_df = pd.read_csv(
        args.abc_data_path, sep="\t", compression="gzip", header=None, low_memory=False,
    )
    ncols = abc_df.shape[1]
    print(f"[INFO] ABC data has {ncols} columns")
    abc_df = abc_df.rename(columns={
        0: "chr", 1: "start", 2: "end", 3: "name", 4: "class", 5: "score_col5", 6: "TargetGene"
    })
    abc_df["cell_type"] = abc_df.iloc[:, -1].astype(str)

    abc_score_candidates = []
    for col_idx in range(max(15, ncols - 5), ncols - 1):
        col_data = pd.to_numeric(abc_df.iloc[:, col_idx], errors="coerce")
        if col_data.notna().sum() > 0:
            col_max = col_data.max()
            col_min = col_data.min()
            if 0 <= col_min and col_max <= 2:
                abc_score_candidates.append((col_idx, col_data.mean()))

    if abc_score_candidates:
        best_col = min(abc_score_candidates, key=lambda x: abs(x[1] - 0.1))[0]
        abc_df["ABC.Score"] = pd.to_numeric(abc_df.iloc[:, best_col], errors="coerce")
        print(f"[INFO] Using column {best_col} as ABC.Score (mean={abc_df['ABC.Score'].mean():.4f})")
    else:
        if ncols > 20:
            abc_df["ABC.Score"] = pd.to_numeric(abc_df.iloc[:, 20], errors="coerce")
            print(f"[WARN] Using fallback column 20 as ABC.Score")
        else:
            abc_df["ABC.Score"] = pd.to_numeric(abc_df.iloc[:, -2], errors="coerce")
            print(f"[WARN] Using fallback column {ncols-2} as ABC.Score")
    abc_df["ABC.Score"] = pd.to_numeric(abc_df["ABC.Score"], errors="coerce")

    print(f"[INFO] Filtering ABC score data for cell_type={args.abc_cell_type}...")
    abc_df = abc_df[abc_df["cell_type"] == args.abc_cell_type].copy()
    abc_df = filter_primary_chroms_df(abc_df, chr_col="chr")
    print(f"[INFO] ABC score data loaded: {len(abc_df)} rows")

    print("[INFO] Loading fanta.bio data...")
    cre_df = pd.read_csv(
        args.cre_data_path, sep="\t", header=None,
        names=["chr", "start", "end", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb", "blockName",
               "attributes", "gene"]
    )
    cre_df = filter_primary_chroms_df(cre_df, chr_col="chr")
    cre_df["gene"] = cre_df["gene"].replace("NA", pd.NA)
    cre_df_with_genes = cre_df.dropna(subset=["gene"]).copy()
    print(f"[INFO] fanta.bio data loaded: {len(cre_df)} rows (with gene: {len(cre_df_with_genes)})")

    re2g_datasets = {}
    for name, path in [("re2g_extended", args.re2g_extended_data_path),
                        ("re2g", args.re2g_data_path)]:
        print(f"[INFO] Loading {name} data...")
        full_df, with_genes_df = load_re2g_data(path)
        re2g_datasets[name] = {"df": full_df, "with_genes": with_genes_df}
        print(f"[INFO] {name} data loaded: {len(full_df)} rows (with gene: {len(with_genes_df)})")

    for pert in tfs:
        if pert not in df3.columns:
            print(f"[WARN] {pert} not found in df3 columns. Skipping.")
            continue

        tf_symbol = pert.split(".")[-1]

        genes = (
            df3.abs()
               .sort_values(pert, ascending=False)[pert]
               .head(200)
               .index
               .to_list()
        )
        top_genes = set(genes)

        outdir = f"attribution/{study}/{pert}"
        os.makedirs(outdir, exist_ok=True)
        gene_list_path = os.path.join(outdir, f"{pert}_top200_genes.txt")
        with open(gene_list_path, "w") as f:
            f.write("\n".join(genes))

        base_df = pd.read_csv(
            base_bed, sep="\t", header=None,
            names=["chr", "start", "end", "gene", "score", "strand", "split"]
        )
        base_df = base_df[base_df["gene"].isin(genes)].copy()
        base_df = filter_primary_chroms_df(base_df, chr_col="chr")
        if len(base_df) == 0:
            print(f"[WARN] No genes found in base_bed for {pert}. Skipping.")
            continue

        base_df[["win_start", "win_end"]] = base_df.apply(
            lambda row: expand_to_tss_window(row, flank=TSS_WINDOW_FLANK), axis=1
        )
        win_df = base_df[["chr", "win_start", "win_end", "gene"]].copy()
        win_df.columns = ["chr", "start", "end", "gene"]
        win_df = ensure_bed3_safe_df(win_df)
        gene_windows = {r["gene"]: (r["chr"], int(r["start"]), int(r["end"])) for _, r in win_df.iterrows()}
        gene_windows = clip_gene_windows(gene_windows, chrom_sizes)

        pert_seed = (abs(hash(pert)) % 10_000_000) + GENEWISE_SHUFFLE_SEED_BASE

        bed_configs = [
            {
                "name": "attribution", "label": "Attribution",
                "bed_path": f"cre/{study}/{pert}/attribution_{pert}.bed",
                "bed_path_raw": f"attribution/{study}/{pert}/{pert}_peaks.bed",
                "create_from": "attribution_parse", "do_gene_shuffle": True,
            },
            {
                "name": "re2g_extended", "label": "rE2G extended",
                "bed_path": f"cre/{study}/{pert}/re2g_extended_{pert}.bed",
                "create_from": "gene_filtered", "source_key": "re2g_extended",
                "out_cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
                "do_gene_shuffle": True,
            },
            {
                "name": "re2g", "label": "rE2G",
                "bed_path": f"cre/{study}/{pert}/re2g_{pert}.bed",
                "create_from": "gene_filtered", "source_key": "re2g",
                "out_cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
                "do_gene_shuffle": True,
            },
            {
                "name": "abc_score", "label": "ABC score",
                "bed_path": f"cre/{study}/{pert}/abc_score_{pert}.bed",
                "create_from": "abc", "do_gene_shuffle": True,
            },
            {
                "name": "attribution_re2g_extended", "label": "Attribution + rE2G extended",
                "bed_path": f"cre/{study}/{pert}/attribution_re2g_extended_{pert}.bed",
                "create_from": "merge", "do_gene_shuffle": True,
                "source_attribution": f"cre/{study}/{pert}/attribution_{pert}.bed",
                "source_other": f"cre/{study}/{pert}/re2g_extended_{pert}.bed",
                "source_other_type": "re2g_extended",
            },
            {
                "name": "attribution_re2g", "label": "Attribution + rE2G",
                "bed_path": f"cre/{study}/{pert}/attribution_re2g_{pert}.bed",
                "create_from": "merge", "do_gene_shuffle": True,
                "source_attribution": f"cre/{study}/{pert}/attribution_{pert}.bed",
                "source_other": f"cre/{study}/{pert}/re2g_{pert}.bed",
                "source_other_type": "re2g",
            },
            {
                "name": "attribution_abc", "label": "Attribution + ABC score",
                "bed_path": f"cre/{study}/{pert}/attribution_abc_{pert}.bed",
                "create_from": "merge", "do_gene_shuffle": True,
                "source_attribution": f"cre/{study}/{pert}/attribution_{pert}.bed",
                "source_other": f"cre/{study}/{pert}/abc_score_{pert}.bed",
                "source_other_type": "abc_score",
            },
            {
                "name": "tss_1kbp", "label": "TSS±1kbp",
                "bed_path": f"cre/{study}/{pert}/tss_1kbp_{pert}.bed",
                "create_from": "base_promoter", "do_gene_shuffle": False,
            },
            {
                "name": "fanta_bio", "label": "fanta.bio",
                "bed_path": f"cre/{study}/{pert}/fanta_bio_{pert}.bed",
                "create_from": "gene_filtered", "source_key": "fanta_bio",
                "out_cols": ["chr", "start", "end", "name", "score", "strand", "gene"],
                "do_gene_shuffle": True,
            },
            {
                "name": "chip_seq", "label": "ChIP-seq",
                "bed_path": f"cre/{study}/{pert}/chip_seq_{tf_symbol}.bed",
                "create_from": "chip_curated", "do_gene_shuffle": False,
            },
        ]

        for cfg in bed_configs:
            bed_name = cfg["name"]
            bed_path = cfg["bed_path"]
            create_from = cfg["create_from"]
            do_gene_shuffle = cfg.get("do_gene_shuffle", False)

            print(f"\n[INFO] Creating/Checking BED: {pert} / {cfg['label']}")
            os.makedirs(os.path.dirname(bed_path), exist_ok=True)

            seed = pert_seed + (abs(hash(bed_name)) % 1_000_000)

            if create_from == "attribution_parse":
                raw_path = cfg["bed_path_raw"]
                if not os.path.exists(raw_path):
                    print(f"[WARN] Missing Attribution raw bed: {raw_path} (skip)")
                    continue

                parse_and_write_attribution_bed_for_gimme(
                    tf_peaks_bed=raw_path, out_bed=bed_path,
                    top_genes=top_genes, pert=pert, gene_windows=gene_windows,
                )
                if not os.path.exists(bed_path):
                    print(f"[WARN] Attribution parsed bed not created: {bed_path} (skip shuffle)")
                    continue

                if do_gene_shuffle:
                    _shuffle_attribution_bed(cfg, top_genes, gene_windows, seed, pert)
                continue

            if create_from == "gene_filtered":
                source_key = cfg["source_key"]
                out_cols = cfg["out_cols"]
                if source_key in re2g_datasets:
                    source_df = re2g_datasets[source_key]["with_genes"]
                elif source_key == "fanta_bio":
                    source_df = cre_df_with_genes
                else:
                    print(f"[WARN] Unknown source_key: {source_key}")
                    continue

                ok = _create_gene_filtered_bed(
                    source_df, genes, gene_windows, out_cols, bed_path, cfg["label"],
                )
                if ok and do_gene_shuffle:
                    _shuffle_standard_bed(bed_name, bed_path, gene_windows, seed)
                continue

            if create_from == "abc":
                print(f"[INFO] Creating ABC score bed for {len(genes)} genes (cell_type={args.abc_cell_type})")
                abc_filtered = abc_df[abc_df["TargetGene"].isin(genes)].copy()
                if len(abc_filtered) == 0:
                    print(f"[WARN] No ABC score rows for {pert}. Skipping abc_score.")
                    continue
                abc_filtered = abc_filtered.sort_values("ABC.Score", ascending=False)
                abc_unique = abc_filtered.drop_duplicates(subset=["chr", "start", "end", "TargetGene"])

                bed_output = abc_unique[["chr", "start", "end", "TargetGene", "ABC.Score"]].copy()
                bed_output.columns = ["chr", "start", "end", "gene", "score"]
                bed_output = ensure_bed3_safe_df(bed_output)
                bed_output = filter_peaks_to_tss_window(bed_output, gene_windows)
                if len(bed_output) == 0:
                    print(f"[WARN] No ABC score regions in TSS windows for {pert}. Skipping.")
                    continue

                bed_output.to_csv(bed_path, sep="\t", header=False, index=False)
                print(f"[INFO] Wrote: {bed_path} ({len(bed_output)} regions; original rows {len(abc_filtered)})")

                if do_gene_shuffle:
                    _shuffle_standard_bed(bed_name, bed_path, gene_windows, seed)
                continue

            if create_from == "merge":
                ok = _create_merged_bed(
                    cfg["source_attribution"], "attribution",
                    cfg["source_other"], cfg["source_other_type"],
                    bed_path, cfg["label"],
                )
                if ok and do_gene_shuffle:
                    _shuffle_merged_bed(
                        cfg,
                        "source_attribution", "attribution",
                        "source_other", cfg["source_other_type"],
                    )
                continue

            if create_from == "base_promoter":
                print(f"[INFO] Creating TSS±1kbp bed from {base_bed}")
                bed_filtered = base_df.copy()
                bed_filtered[["start", "end"]] = bed_filtered.apply(expand_to_promoter, axis=1)
                bed_filtered = filter_primary_chroms_df(bed_filtered, chr_col="chr")
                bed_filtered = bed_filtered[
                    pd.to_numeric(bed_filtered["end"], errors="coerce") >
                    pd.to_numeric(bed_filtered["start"], errors="coerce")
                ].copy()
                bed_filtered.to_csv(bed_path, sep="\t", header=False, index=False)
                print(f"[INFO] Wrote: {bed_path} ({len(bed_filtered)} regions)")
                continue

            if create_from == "chip_curated":
                print(f"[INFO] Creating curated ChIP-seq bed for TF={tf_symbol} using TSS±{TSS_WINDOW_FLANK}bp windows")
                try:
                    chip_union = load_chip_union_for_tf(
                        chipatlas_dir=args.chipatlas_dir,
                        peak_list_tsv=args.chipatlas_list,
                        tf=tf_symbol,
                        pval_suffix=args.chip_pval_suffix
                    )
                except Exception as e:
                    print(f"[WARN] No ChIP-seq peaks for TF={tf_symbol} ({pert}): {e}")
                    continue

                windows_bt = pybedtools.BedTool.from_dataframe(win_df[["chr", "start", "end", "gene"]]).sort()
                curated = chip_union.intersect(windows_bt, u=True).sort().merge()
                curated = filter_primary_chroms_bt(curated)

                if len(curated) == 0:
                    print(f"[WARN] Curated ChIP-seq peaks became empty for TF={tf_symbol} ({pert}).")
                    continue
                curated.saveas(bed_path)
                print(f"[INFO] Wrote: {bed_path} ({len(curated)} peaks kept after gene-window curation)")
                continue

            raise ValueError(f"Unknown create_from: {create_from}")


if __name__ == "__main__":
    main()
