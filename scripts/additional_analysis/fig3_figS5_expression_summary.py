#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _common import read_tsv, unique, save, output_dir, provenance, bootstrap_mean


def row_correlation(x, y):
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    result = np.full(len(x), np.nan)
    valid &= denom > 0
    result[valid] = (xc[valid] * yc[valid]).sum(axis=1) / denom[valid]
    return np.clip(result, -1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="TSV; paths relative to this manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    manifest = read_tsv(a.manifest, ["study", "backbone", "fold", "observed", "prediction", "control"])
    unique(manifest, ["study", "backbone", "fold"])
    out = output_dir(a.out)
    inputs, bins, checkpoints, genes, conditions = [a.manifest], [], [], [], []
    for row in manifest.to_dict("records"):
        paths = {k: Path(a.manifest).resolve().parent / row[k] for k in ["observed", "prediction"]}
        inputs.extend(paths.values())
        obs = pd.read_csv(paths["observed"], sep="\t", index_col=0)
        if "training" not in obs:
            raise ValueError("Observed matrix requires training column")
        locus_ids = obs.index.astype(str).to_numpy()
        if pd.notna(row.get("bed")):
            bed_path = Path(a.manifest).resolve().parent / row["bed"]
            inputs.append(bed_path)
            bed = pd.read_csv(bed_path, sep="\t", header=None,
                              names=["chr", "start", "end", "gene", "score", "strand", "split"])
            if len(bed) != len(obs) or not np.array_equal(bed.gene.astype(str), obs.index.astype(str)):
                raise ValueError("BED and observed matrix gene row order differs")
            if not np.array_equal(bed.split.astype(str), obs.training.astype(str)):
                raise ValueError("BED and observed split row order differs")
            locus_ids = (bed.chr + ":" + bed.start.astype(str) + "-" + bed.end.astype(str)
                         + ":" + bed.strand + ":" + bed.gene).to_numpy()
            if pd.Index(locus_ids).has_duplicates:
                raise ValueError("Duplicate full locus identifiers")
        elif obs.index.has_duplicates:
            raise ValueError("Duplicate gene symbols require an aligned BED manifest column")
        cols = obs.columns.drop("training")
        if row["control"] not in cols or len(cols) < 2:
            raise ValueError("Control must identify an observed channel; at least two channels required")
        pred = np.load(paths["prediction"], allow_pickle=False)
        x = obs[cols].to_numpy(dtype=float)
        if pred.shape != x.shape:
            raise ValueError(f"Prediction/observed shape mismatch: {pred.shape} vs {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observed expression prevents expression-quintile definition")
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        quintile, edges = pd.qcut(x.mean(axis=1), 5, labels=labels, retbins=True)
        test = obs.training.eq("test").to_numpy()
        r = row_correlation(x, pred)
        info = {k: row[k] for k in ["study", "backbone", "fold"]}
        count = {"n_conditions": len(cols), "n_noncontrol": len(cols)-1, "n_control": 1}
        for i, label in enumerate(labels):
            mask = test & (quintile == label)
            finite = r[mask & np.isfinite(r)]
            quant = np.quantile(finite, [.25, .5, .75]) if len(finite) else [np.nan]*3
            bins.append(dict(info, **count, quintile=label, lower=edges[i], upper=edges[i+1],
                             n_test=int(mask.sum()), n_finite=len(finite), n_undefined=int(mask.sum())-len(finite),
                             q1=quant[0], median=quant[1], q3=quant[2]))
        genes.append(pd.DataFrame({**info, "gene": obs.index, "locus_id": locus_ids,
                                   "row_index": np.arange(len(obs)), "training": obs.training.to_numpy(),
                                   "quintile": quintile, "r": r}))
        cr = row_correlation(x[test].T, pred[test].T)
        conditions.append(pd.DataFrame({**info, "condition": cols, "r": cr}))
        finite = cr[np.isfinite(cr)]
        lo, hi = bootstrap_mean(finite, a.seed, a.bootstrap)
        import hashlib
        gene_hash = hashlib.sha256("\n".join(sorted(obs.index[test].astype(str))).encode()).hexdigest()
        locus_hash = hashlib.sha256("\n".join(sorted(locus_ids[test])).encode()).hexdigest()
        gene_values = r[test & np.isfinite(r)]
        gq = np.quantile(gene_values, [.25, .5, .75]) if len(gene_values) else [np.nan]*3
        checkpoints.append(dict(info, **count, n_test_genes=int(test.sum()), test_gene_sha256=gene_hash,
                                test_locus_sha256=locus_hash, n_unique_test_symbols=obs.index[test].nunique(),
                                gene_r_q1=gq[0], gene_r_median=gq[1], gene_r_q3=gq[2],
                                n_finite_conditions=len(finite), n_undefined_conditions=len(cr)-len(finite),
                                mean_r=finite.mean() if len(finite) else np.nan, ci_low=lo, ci_high=hi))
    save(pd.DataFrame(bins), out, "expression_quintile_summary.tsv")
    summary = pd.DataFrame(checkpoints)
    save(summary, out, "fold_checkpoint_summary.tsv")
    save(summary, out, "backbone_comparison_summary.tsv")
    save(pd.concat(genes), out, "gene_correlations.tsv")
    save(pd.concat(conditions), out, "condition_correlations.tsv")
    provenance(out, a, inputs)


if __name__ == "__main__":
    main()
