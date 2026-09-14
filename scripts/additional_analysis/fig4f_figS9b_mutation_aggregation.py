#!/usr/bin/env python
"""A5 / R1 M5,m7–8; R2 M7,m2; R3 2–3: matched mutation effects."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from _common import read_tsv, unique, save, output_dir, provenance, bootstrap_mean, text


def effects(wt, mt, ctrl, channel):
    w, m = float(wt[channel]-wt[ctrl]), float(mt[channel]-mt[ctrl])
    return dict(wt_fc=w, mutant_fc=m, control_effect=float(mt[ctrl]-wt[ctrl]),
                pert_effect=float(mt[channel]-wt[channel]), delta_delta=m-w,
                abs_delta_delta=abs(m-w), magnitude_attenuation=abs(w)-abs(m),
                attenuation_fraction=(abs(w)-abs(m))/abs(w) if w != 0 else np.nan,
                sign_reversal=bool(w*m < 0), zero_effect=bool(m == w))


# The motif and control magnitudes are carried through the same two-stage
# aggregation so that the paired difference can be reported against the scale it
# was taken on. Because the gene stage is a median, the aggregated difference is
# not the difference of the aggregated magnitudes; paired_difference remains the
# quantity the summary and its bootstrap interval are computed from.
AGGREGATED = ["paired_difference", "abs_delta_delta", "abs_delta_delta_control"]

TERTILES = ["Low", "Mid", "High"]


def aggregate(pairs, attr_group="High"):
    selected = pairs[pairs.attr_group == attr_group]
    gene = selected.groupby(["gene", "source_perturbation"], as_index=False)[AGGREGATED].median()
    pert = gene.groupby("source_perturbation", as_index=False)[AGGREGATED].mean()
    return gene, pert


def aggregate_tertiles(pairs, tertiles=TERTILES):
    """Run the same two-stage aggregation separately in each attribution tertile.

    The tertile axis is descriptive. It shows how the paired motif-minus-control
    difference tracks attribution level after sites and output channels have been
    collapsed to one value per perturbation; it does not make attribution and the
    finite mutation effect independent of each other.
    """
    genes, perts = [], []
    for group in tertiles:
        gene, pert = aggregate(pairs, group)
        genes.append(gene.assign(attr_group=group))
        perts.append(pert.assign(attr_group=group))
    columns = ["attr_group"] + [c for c in genes[0].columns if c != "attr_group"]
    return (pd.concat(genes, ignore_index=True)[columns],
            pd.concat(perts, ignore_index=True)[["attr_group", "source_perturbation"] + AGGREGATED])


def summarise(values, seed, bootstrap):
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_mean(values, seed, bootstrap)
    nonzero = values[values != 0]
    return dict(n_perturbations=len(values),
                mean_paired_difference=values.mean() if len(values) else np.nan,
                ci_low=lo, ci_high=hi, n_nonzero_perturbations=len(nonzero),
                sign_test_two_sided=binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tss-bed", required=True, help="Exact inference BED, used to detect out-of-window WT placeholders")
    p.add_argument("--context-length", required=True, type=int)
    p.add_argument("--low-cutoff", required=True, type=float)
    p.add_argument("--high-cutoff", required=True, type=float)
    p.add_argument("--cutoff-source", required=True, help="Record of original attribution tertile boundaries")
    p.add_argument("--expected-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not np.isfinite([a.low_cutoff, a.high_cutoff]).all() or a.low_cutoff >= a.high_cutoff:
        p.error("Finite ordered original cutoffs are required")
    if a.context_length <= 0 or a.context_length % 2 or a.expected_seeds < 1:
        p.error("Invalid context length or seed count")
    root = Path(a.data_dir)
    attr = read_tsv(root / "seqlet_attribution.tsv", ["site_id", "perturbation", "attr_sum_abs"])
    # Multiple annotated seqlets may refer to one site/channel. Never silently take the last value.
    attr = attr[["site_id", "perturbation", "attr_sum_abs"]].drop_duplicates()
    unique(attr, ["site_id", "perturbation"])
    targets = read_tsv(root / "mutation_targets.tsv", ["gene", "site_id", "target_id", "mutation_type", "mut_start", "mut_end", "chromosome"])
    unique(targets, ["target_id"])
    bed = pd.read_csv(a.tss_bed, sep="\t", header=None,
                      names=["chr", "start", "end", "gene", "score", "strand", "split"])
    if bed.gene.isna().any():
        raise ValueError("Missing inference BED gene identifiers")
    # Reproduce 52b: first entry matching any target chromosome, else first entry.
    chosen = []
    for gene, gt in targets.groupby("gene", sort=True):
        entries = bed[bed.gene == gene]
        if entries.empty:
            continue
        matching = entries[entries.chr.isin(gt.chromosome.unique())]
        chosen.append((matching if len(matching) else entries).iloc[0])
    duplicate_bed = bed[bed.duplicated("gene", keep=False)].copy()
    bed = pd.DataFrame(chosen, columns=bed.columns)
    # Reproduce 52b get_gene_info and HALF_CONTEXT exactly, including its
    # BED-start convention on the minus strand (not 43's distance convention).
    bed["tss"] = bed.start
    bed = bed.set_index("gene")
    out = output_dir(a.out)
    save(duplicate_bed, out, "duplicate_inference_loci_audit.tsv")
    save(bed.reset_index(), out, "selected_inference_loci.tsv")
    inputs = [root / "seqlet_attribution.tsv", root / "mutation_targets.tsv", a.tss_bed]
    rows, audit = [], []
    for gene, gt in targets.groupby("gene", sort=True):
        path = root / "results" / f"{gene}_predictions.h5"
        inputs.append(path)
        try:
            with h5py.File(path, "r") as hf:
                wt, mt = hf["wt_pred"][:], hf["mt_pred_mean"][:]
                meta = hf["metadata"]
                cols = [text(x) for x in meta["expr_cols"][:]]
                ids = [text(x) for x in meta["target_ids"][:]]
                saved_sites = [text(x) for x in meta["site_ids"][:]]
                saved_types = [text(x) for x in meta["mutation_types"][:]]
                saved_chroms = [text(x) for x in meta["chromosomes"][:]]
                saved_starts, saved_ends = meta["mut_starts"][:], meta["mut_ends"][:]
                ctrl = cols.index(text(meta["control_col"][()]))
                if text(meta["gene"][()]) != gene or len(set(ids)) != len(ids) or len(set(cols)) != len(cols):
                    raise ValueError("Duplicate IDs or incorrect gene metadata")
                if wt.shape != (len(cols),) or mt.shape != (len(ids), len(cols)):
                    raise ValueError("Incomplete prediction shapes")
                if any(len(v) != len(ids) for v in [saved_sites, saved_types, saved_chroms, saved_starts, saved_ends]):
                    raise ValueError("Incomplete target metadata")
                if int(meta["n_seeds"][()]) != a.expected_seeds:
                    raise ValueError("Unexpected seed count")
                if "mt_pred" in hf:
                    seeds = hf["mt_pred"][:]
                    if seeds.shape != (len(ids), a.expected_seeds, len(cols)):
                        raise ValueError("Incomplete seed predictions")
                    if not np.allclose(seeds.mean(axis=1), mt, equal_nan=True):
                        raise ValueError("Saved mean differs from seed mean")
        except (OSError, KeyError, ValueError) as error:
            for target in gt.itertuples():
                audit.append(dict(gene=gene, target_id=target.target_id, reason=f"unreadable_or_incomplete_h5: {error}"))
            continue
        for target in gt.itertuples():
            reason = ""
            if target.target_id not in ids:
                reason = "missing_target_prediction"
            elif gene not in bed.index:
                reason = "missing_inference_tss"
            else:
                locus = bed.loc[gene]
                start = locus.tss-(a.context_length-1)//2
                if (target.chromosome != locus.chr or target.mut_start < start or
                        target.mut_end > start+a.context_length or target.mut_end <= target.mut_start):
                    reason = "invalid_mutation_window_WT_placeholder"
            if reason:
                audit.append(dict(gene=gene, target_id=target.target_id, reason=reason))
                continue
            ti = ids.index(target.target_id)
            if (saved_sites[ti], saved_types[ti], saved_chroms[ti], saved_starts[ti], saved_ends[ti]) != (
                    target.site_id, target.mutation_type, target.chromosome, target.mut_start, target.mut_end):
                audit.append(dict(gene=gene, target_id=target.target_id, reason="target_metadata_mismatch"))
                continue
            if not np.isfinite(wt).all() or not np.isfinite(mt[ti]).all():
                audit.append(dict(gene=gene, target_id=target.target_id, reason="nonfinite_predictions"))
                continue
            sources = attr[attr.site_id == target.site_id]
            if sources.empty:
                audit.append(dict(gene=gene, target_id=target.target_id, reason="missing_source_attribution"))
            for source in sources.itertuples():
                if source.perturbation not in cols or source.perturbation == cols[ctrl] or not np.isfinite(source.attr_sum_abs):
                    audit.append(dict(gene=gene, target_id=target.target_id, reason="invalid_source_channel_or_attribution"))
                    continue
                group = "Low" if source.attr_sum_abs <= a.low_cutoff else "Mid" if source.attr_sum_abs <= a.high_cutoff else "High"
                rows.append(dict(gene=gene, site_id=target.site_id, target_id=target.target_id,
                                 mutation_type=target.mutation_type, source_perturbation=source.perturbation,
                                 output_channel=source.perturbation, control_channel=cols[ctrl],
                                 attr_group=group, attr_sum_abs=source.attr_sum_abs,
                                 **effects(wt, mt[ti], ctrl, cols.index(source.perturbation))))
    save(pd.DataFrame(audit, columns=["gene", "target_id", "reason"]), out, "mutation_exclusions.tsv")
    provenance(out, a, inputs)
    if not rows:
        raise RuntimeError(f"No valid observations; see {out}/mutation_exclusions.tsv")
    long = pd.DataFrame(rows)
    keys = ["gene", "site_id", "source_perturbation", "output_channel"]
    unique(long, keys + ["mutation_type"])
    motif = long[long.mutation_type == "seqlet_mut"]
    controls = long[long.mutation_type == "neg_control"]
    paired = motif.merge(controls[keys+["target_id", "abs_delta_delta"]], on=keys, how="left",
                         suffixes=("", "_control"), validate="one_to_one")
    paired["paired_complete"] = paired.target_id_control.notna()
    paired["paired_difference"] = paired.abs_delta_delta-paired.abs_delta_delta_control
    complete = paired[paired.paired_complete]
    gene, pert = aggregate(complete)
    summary = summarise(pert.paired_difference.to_numpy(), a.seed, a.bootstrap)
    tertile_gene, tertile_pert = aggregate_tertiles(complete)
    tertile_summary = pd.DataFrame([
        dict(attr_group=group,
             **summarise(tertile_pert.paired_difference[tertile_pert.attr_group == group].to_numpy(),
                         a.seed, a.bootstrap),
             mean_abs_delta_delta=tertile_pert.abs_delta_delta[tertile_pert.attr_group == group].mean(),
             mean_abs_delta_delta_control=tertile_pert.abs_delta_delta_control[tertile_pert.attr_group == group].mean())
        for group in TERTILES])
    counts = dict(n_genes=long.gene.nunique(), n_unique_sites=long[["gene", "site_id"]].drop_duplicates().shape[0],
                  n_mutation_targets=long.target_id.nunique(), n_source_perturbations=long.source_perturbation.nunique(),
                  n_output_channels=long.output_channel.nunique(), n_site_channel_rows=len(long),
                  n_paired_complete=int(paired.paired_complete.sum()), n_zero_effect_rows=int(long.zero_effect.sum()),
                  n_audit_rows=len(audit))
    for df, name in [(long, "mutation_long.tsv"), (paired, "mutation_pairs.tsv"), (gene, "mutation_gene_effects.tsv"),
                     (pert, "mutation_perturbation_effects.tsv"), (pd.DataFrame([summary]), "mutation_summary.tsv"),
                     (tertile_gene, "mutation_tertile_gene_effects.tsv"),
                     (tertile_pert, "mutation_tertile_perturbation_effects.tsv"),
                     (tertile_summary, "mutation_tertile_summary.tsv"),
                     (pd.DataFrame([counts]), "mutation_sample_sizes.tsv")]:
        save(df, out, name)


if __name__ == "__main__":
    main()
