#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from _common import read_tsv, unique, save, output_dir, provenance, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attribution_evaluation"))
from _gene_aware_intervals import merge_intervals_by_gene, assign_max_overlap_scores_by_gene

STRATA = [("promoter_0_1kb", 0, 1000), ("proximal_1_10kb", 1000, 10000),
          ("distal_10_100kb", 10000, 100000), ("very_distal_100kb", 100000, np.inf)]
METHODS = ["GenPerturb", "ABC", "rE2G", "rE2G_extended", "TSS_distance"]


def bed(path, gene_col=None):
    # An existing empty BED means no annotated candidates; a missing file is an error.
    try:
        raw = pd.read_csv(path, sep="\t", header=None, comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["chr", "start", "end", "gene", "score"])
    out = raw.iloc[:, :3].copy()
    out.columns = ["chr", "start", "end"]
    out["chr"] = out.chr.astype(str)
    for c in ["start", "end"]:
        values = pd.to_numeric(out[c], errors="raise")
        if (values % 1 != 0).any():
            raise ValueError(f"Noninteger BED coordinates: {path}")
        out[c] = values.astype(int)
    if ((out.start < 0) | (out.end <= out.start)).any():
        raise ValueError(f"Invalid BED intervals: {path}")
    if gene_col is not None:
        out["gene"] = raw.iloc[:, gene_col].astype(str)
        out["score"] = pd.to_numeric(raw.iloc[:, 4], errors="raise")
        if not np.isfinite(out.score).all():
            raise ValueError(f"Nonfinite annotation score: {path}")
    return out


def build_atac_index(truth):
    """Index half-open ATAC intervals by chromosome, independent of candidates.

    Prefix maximum ends handle nested intervals as well as overlapping ones.
    Build once per perturbation and reuse for both candidate universes.
    """
    index = {}
    for chrom, group in truth.groupby("chr", sort=False):
        starts = group.start.to_numpy(dtype=np.int64)
        ends = group.end.to_numpy(dtype=np.int64)
        order = np.argsort(starts, kind="stable")
        index[str(chrom)] = (starts[order], np.maximum.accumulate(ends[order]))
    return index


def atac_overlap_labels(peaks, index):
    """Return any-overlap labels in row order without candidate × ATAC scans.

    For [start,end), restrict ATAC starts to < end using binary search.
    An overlap exists iff the maximum end in that prefix is > start.
    Endpoint contact alone is therefore not an overlap, matching BED semantics.
    """
    labels = np.zeros(len(peaks), dtype=np.int8)
    for chrom, positions in peaks.groupby("chr", sort=False).indices.items():
        if str(chrom) not in index:
            continue
        starts, max_ends = index[str(chrom)]
        queries = peaks.iloc[positions]
        right = np.searchsorted(starts, queries.end.to_numpy(dtype=np.int64), side="left")
        valid = right > 0
        labels[positions[valid]] = (
            max_ends[right[valid]-1] > queries.start.to_numpy(dtype=np.int64)[valid]
        )
    return labels


def raw_scores(peaks, path):
    scores = np.full(len(peaks), np.nan)
    reasons = np.full(len(peaks), "", dtype=object)
    if not Path(path).is_file():
        return scores, np.full(len(peaks), "missing_h5", dtype=object)
    with h5py.File(path, "r") as hf:
        for gene, group in peaks.groupby("gene"):
            if gene not in hf or "ixg_fc" not in hf[gene]:
                reasons[group.index] = "missing_gene_or_ixg_fc"
                continue
            g = hf[gene]
            if "chromosome" not in g.attrs or "seq_start" not in g.attrs:
                reasons[group.index] = "missing_coordinate_metadata"
                continue
            arr = g["ixg_fc"]
            if arr.ndim != 2 or arr.shape[1] != 4:
                reasons[group.index] = "invalid_ixg_shape"
                continue
            start = int(g.attrs["seq_start"])
            for i, row in group.iterrows():
                lo, hi = max(row.start, start), min(row.end, start + arr.shape[0])
                if row.chr != text(g.attrs["chromosome"]) or lo >= hi:
                    reasons[i] = "no_coordinate_overlap"
                    continue
                values = np.abs(arr[lo-start:hi-start, :]).sum(axis=1)
                if not np.isfinite(values).all():
                    reasons[i] = "nonfinite_ixg"
                    continue
                n = max(1, int(np.ceil(len(values) * .1)))
                scores[i] = np.partition(values, len(values)-n)[-n:].mean()
    return scores, reasons


def evaluate(peaks, pert, universe, min_positive):
    rows = []
    for name, lo, hi in STRATA:
        sub = peaks[(peaks.distance >= lo) & (peaks.distance < hi)]
        n, pos = len(sub), int(sub.positive.sum())
        reason = ""
        if pos < min_positive:
            reason = "positive_below_minimum"
        elif pos == n:
            reason = "no_negatives"
        elif not np.isfinite(sub[METHODS].to_numpy()).all():
            reason = "missing_method_scores"
        for method in METHODS:
            rows.append(dict(universe=universe, perturbation=pert, stratum=name, method=method,
                             n_candidates=n, n_positive=pos, n_negative=n-pos,
                             prevalence=pos/n if n else np.nan, exclusion_reason=reason,
                             auprc=np.nan if reason else average_precision_score(sub.positive, sub[method])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="TSV: perturbation, cre_dir, raw_h5, atac_bed; optional pipeline_union")
    p.add_argument("--genes", required=True, help="Table S3 xlsx or TSV: perturbation,gene,log2FC")
    p.add_argument("--tss-bed", required=True)
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--context-length", type=int, required=True)
    p.add_argument("--min-positive", type=int, default=10)
    p.add_argument("--fc-min", type=float, default=.5)
    p.add_argument(
        "--missing-attribution-policy", choices=["exclude", "zero"], default="exclude",
        help=("How to handle candidate intervals whose raw GenPerturb attribution is "
              "unavailable. 'exclude' excludes the complete stratum; 'zero' retains "
              "the interval with a zero GenPerturb score."),
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.context_length <= 0 or a.context_length % 2 or a.min_positive < 1 or a.fc_min < 0:
        p.error("Invalid context length or threshold")
    manifest = read_tsv(a.manifest, ["perturbation", "cre_dir", "raw_h5", "atac_bed"])
    unique(manifest, ["perturbation"])
    if Path(a.genes).suffix == ".xlsx":
        genes = pd.read_excel(a.genes, sheet_name="TF_sensitive_genes", header=2).rename(
            columns={"perturbation_name": "perturbation", "gene_ID": "gene"})
    else:
        genes = read_tsv(a.genes, ["perturbation", "gene", "log2FC"])
    genes = genes[genes.log2FC.abs() >= a.fc_min]
    tss = pd.read_csv(a.tss_bed, sep="\t", header=None,
                      names=["chr", "start", "end", "gene", "score", "strand", "split"])
    if tss.gene.isna().any():
        raise ValueError("Missing TSS gene identifiers")
    tss["tss"] = np.where(tss.strand == "+", tss.start, tss.end)
    sizes = pd.read_csv(a.chrom_sizes, sep="\t", header=None, names=["chr", "size"]).set_index("chr")["size"]
    primary = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    tss = tss[tss.chr.isin(primary)].copy()
    # Repeated gene identifiers resolve to the final BED locus, matching the
    # dictionaries used by candidate generation and peak scoring.
    duplicate_tss = tss[tss.duplicated("gene", keep=False)].copy()
    tss = tss.drop_duplicates("gene", keep="last")
    if not tss.chr.isin(sizes.index).all():
        raise ValueError("Chromosome sizes missing for TSS chromosomes")
    out = output_dir(a.out)
    save(duplicate_tss, out, "duplicate_tss_last_locus_audit.tsv")
    inputs, results, audits = [a.manifest, a.genes, a.tss_bed, a.chrom_sizes], [], []
    for row in manifest.to_dict("records"):
        pert = row["perturbation"]
        if "/" in pert or pert in [".", ".."]:
            raise ValueError("Invalid perturbation filename")
        resolve = lambda value: Path(a.manifest).resolve().parent / value
        cre, raw, atac = [resolve(row[k]) for k in ["cre_dir", "raw_h5", "atac_bed"]]
        inputs.extend([raw, atac])
        keep = set(genes.loc[genes.perturbation.isin([pert, pert.split(".")[-1]]), "gene"])
        base = tss[tss.gene.isin(keep)].copy()
        sources = {}
        for method, prefix, gc in [("ABC", "abc_score", 3), ("rE2G", "re2g", 6), ("rE2G_extended", "re2g_extended", 6)]:
            path = cre / f"{prefix}_{pert}.bed"
            inputs.append(path)
            frame = bed(path, gc).merge(base[["gene", "chr", "tss"]], on=["gene", "chr"], validate="many_to_one")
            # Retain intervals overlapping the clipped TSS window without
            # changing their annotation boundaries.
            win_start = np.maximum(0, frame.tss-a.context_length//2)
            win_end = np.minimum(frame.chr.map(sizes), frame.tss+a.context_length//2)
            sources[method] = frame[(frame.end > win_start) & (frame.start < win_end)].assign(source=method)
        promoter = base[["chr", "gene"]].copy()
        promoter["start"] = np.maximum(0, base.tss-1000)
        promoter["end"] = np.minimum(base.chr.map(sizes), base.tss+1000)
        promoter["score"], promoter["source"] = 0., "TSS_1kb"
        candidates = merge_intervals_by_gene(pd.concat([*sources.values(), promoter], ignore_index=True))
        universes = {"independent": candidates}
        if pd.notna(row.get("pipeline_union")):
            path = resolve(row["pipeline_union"])
            inputs.append(path)
            pipeline = pd.read_csv(path, sep="\t", header=None,
                                   names=["chr", "start", "end", "gene", "sources", "max_score", "pipeline_positive"])
            universes["pipeline"] = pipeline[pipeline.gene.isin(keep)].drop(columns="pipeline_positive")
        atac_index = build_atac_index(bed(atac))
        for universe, peaks in universes.items():
            peaks = peaks.merge(base[["gene", "chr", "tss"]], on=["gene", "chr"], validate="many_to_one").reset_index(drop=True)
            unique(peaks, ["chr", "start", "end", "gene"])
            peaks["distance"] = abs((peaks.start+peaks.end)/2-peaks.tss)
            peaks["TSS_distance"] = 1/(1+peaks.distance)
            peaks["positive"] = atac_overlap_labels(peaks, atac_index)
            for method, source in sources.items():
                peaks[method] = assign_max_overlap_scores_by_gene(peaks, source)
            peaks["GenPerturb"], peaks["missing_reason"] = raw_scores(peaks, raw)
            n_missing = int(peaks.GenPerturb.isna().sum())
            if a.missing_attribution_policy == "zero":
                peaks["GenPerturb"] = peaks.GenPerturb.fillna(0.0)
            save(peaks, out, f"{pert}_{universe}_scored_candidates.tsv")
            peaks[["chr", "start", "end", "gene", "sources", "max_score", "positive"]].to_csv(
                out / f"{pert}_{universe}.bed", sep="\t", header=False, index=False)
            results.extend(evaluate(peaks, pert, universe, a.min_positive))
            audits.append(dict(perturbation=pert, universe=universe, n_deg=len(keep),
                               n_deg_with_tss=len(base), n_candidates=len(peaks),
                               n_missing_scores=n_missing,
                               missing_attribution_policy=a.missing_attribution_policy))
    save(pd.DataFrame(results), out, "auprc_per_perturbation.tsv")
    save(pd.DataFrame(audits), out, "candidate_universe_audit.tsv")
    provenance(out, a, inputs)


if __name__ == "__main__":
    main()
