#!/usr/bin/env python
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
os.chdir(CWD)

_CFG_PATH = os.path.join(CWD, "scripts/attribution_evaluation/52_study_config.py")
_spec = importlib.util.spec_from_file_location("study_config_52", _CFG_PATH)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
STUDY_CONFIGS = _cfg.STUDY_CONFIGS
CONTEXT_LENGTH = _cfg.CONTEXT_LENGTH
HALF_CONTEXT = _cfg.HALF_CONTEXT
RANDOM_SEED = _cfg.RANDOM_SEED
DATA_DIR_TPL = _cfg.DATA_DIR_TPL

SEQLET_MATCHED = "attribution_analysis/captum_union/seqlet_loci_matched_core.tsv"
SEQLET_ALL = "attribution_analysis/captum_union/seqlet_loci_all_core.tsv"

NEG_CONTROL_OFFSETS = [500, -500, 1000, -1000, 2000, -2000, 3000, -3000]


def parse_args():
    p = argparse.ArgumentParser()
    study_choices = [
        key for key, cfg in STUDY_CONFIGS.items()
        if cfg["seqlet_mode"] in {"nonmatched", "full"}
    ]
    p.add_argument("--study", required=True, choices=study_choices)
    p.add_argument("--sample-size", type=int, default=None,
                   help="Number of unique sites to sample "
                        "(default: from study_config)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def load_seqlet_data(study_key):
    matched = pd.read_csv(SEQLET_MATCHED, sep="\t")
    matched = matched[matched["study"] == study_key].copy()

    all_seqlets = pd.read_csv(SEQLET_ALL, sep="\t")
    all_seqlets = all_seqlets[all_seqlets["study"] == study_key].copy()

    print(f"Matched seqlets ({study_key}): {len(matched)}")
    print(f"All seqlets ({study_key}): {len(all_seqlets)}")
    return matched, all_seqlets


def select_nonmatched(all_seqlets, matched):
    key_cols = ["perturbation", "pattern", "seqlet_idx", "gene",
                "chromosome", "genomic_start", "genomic_end"]
    matched_keys = matched[key_cols].drop_duplicates().assign(_matched=1)
    merged = all_seqlets.merge(matched_keys, on=key_cols, how="left")
    nonmatched = merged[merged["_matched"].isna()].drop(columns="_matched").copy()
    nonmatched = nonmatched.dropna(subset=["core_genomic_start", "core_genomic_end"])
    print(f"Non-matched seqlets (with core): {len(nonmatched)}")
    return nonmatched


def rename_best_to_matched(df):
    out = df.copy()
    out["matched_motif"] = out.get("best_match", pd.Series(dtype=object))
    out["matched_qval"] = out.get("best_qval", pd.Series(dtype=float))
    out["matched_motif_gene"] = out.get("best_motif_gene", pd.Series(dtype=object))
    if "gene_match" not in out.columns:
        out["gene_match"] = False
    if "cluster_match" not in out.columns:
        out["cluster_match"] = False
    return out


def combine_full_seqlets(all_seqlets, matched):
    matched = matched.dropna(subset=["core_genomic_start", "core_genomic_end"]).copy()
    nonmatched = rename_best_to_matched(select_nonmatched(all_seqlets, matched))
    combined = pd.concat([matched, nonmatched], ignore_index=True, sort=False)
    print(f"Full seqlets (matched + non-matched, with core): {len(combined)}")
    return combined


def sample_sites(seqlets, sample_size, seed):
    site_cols = ["chromosome", "core_genomic_start", "core_genomic_end", "gene"]
    unique_sites = seqlets.drop_duplicates(subset=site_cols)
    n_unique = len(unique_sites)
    print(f"Unique sites: {n_unique}")

    if sample_size is None or sample_size >= n_unique:
        selected = unique_sites.copy()
        print(f"Using ALL unique sites ({n_unique})")
    else:
        selected = unique_sites.sample(n=sample_size, random_state=seed).copy()
        print(f"Randomly sampled {sample_size} sites (seed={seed})")

    sampled_keys = selected[site_cols].drop_duplicates().assign(_keep=1)
    meta = seqlets.merge(sampled_keys, on=site_cols, how="inner").drop(columns="_keep")
    meta["seqlet_instance_id"] = (
        meta["perturbation"] + "__" + meta["pattern"] + "__"
        + meta["seqlet_idx"].astype(str) + "__" + meta["gene"]
    )
    meta["site_id"] = (
        meta["chromosome"] + ":" + meta["core_genomic_start"].astype(int).astype(str)
        + "-" + meta["core_genomic_end"].astype(int).astype(str) + "__" + meta["gene"]
    )
    print(f"Per-perturbation records retained: {len(meta)}")
    return selected, meta


def build_targets(selected):
    targets = selected[[
        "chromosome", "core_genomic_start", "core_genomic_end", "gene",
    ]].copy()
    targets = targets.rename(columns={
        "core_genomic_start": "mut_start",
        "core_genomic_end": "mut_end",
    })
    targets["mut_length"] = targets["mut_end"] - targets["mut_start"]
    targets["mutation_type"] = "seqlet_mut"
    targets["site_id"] = (
        targets["chromosome"] + ":" + targets["mut_start"].astype(int).astype(str)
        + "-" + targets["mut_end"].astype(int).astype(str) + "__" + targets["gene"]
    )
    targets["target_id"] = targets["site_id"] + "__seqlet"
    print(f"Unique mutation sites: {len(targets)}")
    return targets


def find_negative_controls(targets, all_seqlets, study_name):
    gene_occupied = {}
    for _, row in all_seqlets.iterrows():
        gene = row["gene"]
        if pd.notna(row.get("core_genomic_start")):
            start = int(row["core_genomic_start"]) - 10
            end = int(row["core_genomic_end"]) + 10
        else:
            start = int(row["genomic_start"]) - 10
            end = int(row["genomic_end"]) + 10
        gene_occupied.setdefault(gene, []).append((row["chromosome"], start, end))

    bed_df = pd.read_csv(
        f"fasta/{study_name}.bed",
        sep="\t", header=None,
        names=["chr", "start", "end", "gene", "score", "strand", "split"],
    )
    gene_tss = dict(zip(bed_df["gene"], bed_df["start"]))

    neg_controls = []
    n_found = 0
    n_failed = 0

    for _, row in targets.iterrows():
        gene = row["gene"]
        chrom = row["chromosome"]
        length = int(row["mut_length"])
        center = (int(row["mut_start"]) + int(row["mut_end"])) // 2
        occupied = gene_occupied.get(gene, [])

        tss = gene_tss.get(gene)
        win_start = tss - HALF_CONTEXT if tss is not None else center - HALF_CONTEXT
        win_end = win_start + CONTEXT_LENGTH

        found = False
        for offset in NEG_CONTROL_OFFSETS:
            nc_start = center + offset - length // 2
            nc_end = nc_start + length

            if nc_start < win_start or nc_end > win_end:
                continue

            overlap = False
            for occ_chr, occ_s, occ_e in occupied:
                if occ_chr == chrom and nc_start < occ_e and nc_end > occ_s:
                    overlap = True
                    break
            if not overlap:
                neg_controls.append({
                    "chromosome": chrom,
                    "mut_start": nc_start,
                    "mut_end": nc_end,
                    "gene": gene,
                    "mut_length": length,
                    "mutation_type": "neg_control",
                    "site_id": row["site_id"],
                    "target_id": row["site_id"] + "__negctrl",
                })
                found = True
                n_found += 1
                break

        if not found:
            n_failed += 1

    neg_df = pd.DataFrame(neg_controls)
    print(f"Negative controls found: {n_found}, failed: {n_failed}")
    return neg_df


def extract_attribution(meta, attribution_dir):
    attr_records = []

    for pert, group in meta.groupby("perturbation"):
        h5_path = f"{attribution_dir}/{pert}/{pert}.h5"
        if not os.path.exists(h5_path):
            print(f"  Attribution H5 not found: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            h5_genes = np.array(f["metadata"]["gene"]).astype(str)
            h5_win_starts = np.array(f["metadata"]["window_start"])
            h5_win_ends = np.array(f["metadata"]["window_end"])
            ixg_fc = f["ixg_fc"]

            for _, row in group.iterrows():
                gene = row["gene"]
                core_start = int(row["core_genomic_start"])
                core_end = int(row["core_genomic_end"])
                win_start = int(row["window_start"])
                win_end = int(row["window_end"])

                mask = (
                    (h5_genes == gene)
                    & (h5_win_starts == win_start)
                    & (h5_win_ends == win_end)
                )
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                idx = indices[0]
                window_attr = ixg_fc[idx]

                pos_start = core_start - win_start
                pos_end = core_end - win_start
                if pos_start < 0 or pos_end > 128:
                    continue

                core_attr = window_attr[pos_start:pos_end, :]
                attr_per_pos = core_attr.sum(axis=1)
                attr_abs_per_pos = np.abs(core_attr).sum(axis=1)

                attr_records.append({
                    "seqlet_instance_id": row["seqlet_instance_id"],
                    "site_id": row["site_id"],
                    "perturbation": pert,
                    "attr_sum_signed": float(attr_per_pos.sum()),
                    "attr_sum_abs": float(attr_abs_per_pos.sum()),
                    "attr_mean_signed": float(attr_per_pos.mean()),
                    "attr_mean_abs": float(attr_abs_per_pos.mean()),
                    "attr_max_abs": float(np.abs(attr_per_pos).max()),
                })

        n_pert = sum(1 for r in attr_records if r["perturbation"] == pert)
        print(f"  {pert}: extracted {n_pert} attributions")

    attr_df = pd.DataFrame(attr_records)
    print(f"Total attribution records: {len(attr_df)}")
    return attr_df


def main():
    args = parse_args()
    cfg = STUDY_CONFIGS[args.study]
    study_key = cfg["study_key"]
    seqlet_mode = cfg["seqlet_mode"]
    study_name = cfg["study_name"]
    study_full = cfg["study_full"]
    attribution_dir = f"attribution/{study_full}"

    sample_size = args.sample_size if args.sample_size is not None \
        else cfg["nonmatched_sample_size"]

    outdir = DATA_DIR_TPL.format(study=args.study)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print(f"Preparing {seqlet_mode.upper()} mutation targets for {args.study}")
    print(f"  sample size: {sample_size}, seed: {args.seed}")
    print(f"  output dir : {outdir}")
    print("=" * 60)

    print("\n[Step 1] Loading seqlet data...")
    matched, all_seqlets = load_seqlet_data(study_key)

    if seqlet_mode == "full":
        print("\n[Step 2] Combining matched and non-matched seqlets...")
        seqlets = combine_full_seqlets(all_seqlets, matched)
    else:
        print("\n[Step 2] Selecting non-matched seqlets...")
        seqlets = rename_best_to_matched(select_nonmatched(all_seqlets, matched))

    print("\n[Step 3] Random-sampling unique sites...")
    selected, meta = sample_sites(seqlets, sample_size, args.seed)

    print("\n[Step 4] Building targets table...")
    targets = build_targets(selected)

    print("\n[Step 5] Finding negative control regions...")
    neg_controls = find_negative_controls(targets, all_seqlets, study_name)

    all_targets = pd.concat([targets, neg_controls], ignore_index=True)
    print(f"\nTotal mutation targets: {len(all_targets)}")
    print(f"  Seqlet mutations: {(all_targets['mutation_type'] == 'seqlet_mut').sum()}")
    print(f"  Negative controls: {(all_targets['mutation_type'] == 'neg_control').sum()}")
    print(f"  Unique genes: {all_targets['gene'].nunique()}")

    print("\n[Step 6] Extracting seqlet-level attribution...")
    attr_df = extract_attribution(meta, attribution_dir)

    gene_list = sorted(all_targets["gene"].unique())
    task_file = f"{outdir}/tasks.txt"
    with open(task_file, "w") as fh:
        for gene in gene_list:
            fh.write(f"{gene}\n")
    print(f"\nTask file: {task_file} ({len(gene_list)} genes)")

    targets_path = f"{outdir}/mutation_targets.tsv"
    meta_path = f"{outdir}/seqlet_metadata.tsv"
    attr_path = f"{outdir}/seqlet_attribution.tsv"

    all_targets.to_csv(targets_path, sep="\t", index=False)
    meta.to_csv(meta_path, sep="\t", index=False)
    attr_df.to_csv(attr_path, sep="\t", index=False)

    print(f"\nSaved:")
    print(f"  {targets_path} ({len(all_targets)} rows)")
    print(f"  {meta_path} ({len(meta)} rows)")
    print(f"  {attr_path} ({len(attr_df)} rows)")

    per_gene = all_targets.groupby("gene").size()
    print(f"\n[Summary]")
    print(f"  Targets per gene: mean={per_gene.mean():.1f}, "
          f"median={per_gene.median():.0f}, max={per_gene.max()}")
    print("Done.")


if __name__ == "__main__":
    main()
