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
import torch

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, CWD)
os.chdir(CWD)

_CFG_PATH = os.path.join(CWD, "scripts/attribution_evaluation/52_study_config.py")
_spec = importlib.util.spec_from_file_location("study_config_52", _CFG_PATH)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
STUDY_CONFIGS = _cfg.STUDY_CONFIGS
CONTEXT_LENGTH = _cfg.CONTEXT_LENGTH
HALF_CONTEXT = _cfg.HALF_CONTEXT
FASTA = _cfg.FASTA
DATA_DIR_TPL = _cfg.DATA_DIR_TPL

from genperturb.model._genperturb import GenPerturb
from genperturb.dataloaders._genome import (
    GenomeIntervalDataset,
)
from genperturb.dataloaders._alphagenome_sequence import alphagenome_indices_to_one_hot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=list(STUDY_CONFIGS.keys()))
    p.add_argument("--gene", required=True, help="Target gene symbol")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of random shuffle seeds per target")
    return p.parse_args()


def load_model(study_name, study_full):
    df = pd.read_csv(f"data/{study_name}.tsv", sep="\t", index_col=[0])
    hdf5 = f"data/{study_name}_alphagenome.h5"

    model = GenPerturb(
        df, hdf5=hdf5,
        context_length=CONTEXT_LENGTH,
        pretrained="alphagenome",
        training_method="transfer",
        study=study_full,
        emb_method="tss",
    )
    model.load_model()
    model.load_pretrained_model()
    model.training_method = "prediction"
    model.module.training_method = "prediction"
    model.module.eval()
    model.module.pretrained_model.eval()

    expr_cols = [c for c in df.columns if c != "training"]
    return model, df, expr_cols


def load_genome_dataset(study_name):
    bed = f"fasta/{study_name}.bed"
    return GenomeIntervalDataset(
        bed_file=bed, fasta_file=FASTA,
        return_seq_indices=True, context_length=CONTEXT_LENGTH,
    )


def get_gene_info(study_name):
    bed_df = pd.read_csv(
        f"fasta/{study_name}.bed", sep="\t", header=None,
        names=["chr", "start", "end", "gene", "score", "strand", "split"],
    )
    info = {}
    for idx, row in bed_df.iterrows():
        info.setdefault(row["gene"], []).append({
            "index": idx, "tss": row["start"],
            "chrom": row["chr"], "strand": row["strand"],
        })
    return info


def shuffle_region(seq_indices, start, end, rng):
    mutated = seq_indices.clone()
    region = seq_indices[start:end].numpy().copy()
    rng.shuffle(region)
    mutated[start:end] = torch.from_numpy(region)
    return mutated


def predict_single(model, seq_indices, sequence_key):
    one_hot = alphagenome_indices_to_one_hot(
        seq_indices,
        sequence_key=sequence_key,
    ).unsqueeze(0).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model.module.forward(one_hot, cal_loss=False, prediction=True)
    return pred.cpu().float().numpy().squeeze()


def main():
    args = parse_args()
    cfg = STUDY_CONFIGS[args.study]
    study_name = cfg["study_name"]
    study_full = cfg["study_full"]
    control_col = cfg["control_col"]

    study_dir = DATA_DIR_TPL.format(study=args.study)
    outdir = f"{study_dir}/results"
    os.makedirs(outdir, exist_ok=True)

    targets_file = f"{study_dir}/mutation_targets.tsv"
    targets_df = pd.read_csv(targets_file, sep="\t")
    gene_targets = targets_df[targets_df["gene"] == args.gene].copy()

    if len(gene_targets) == 0:
        print(f"No targets for gene {args.gene}")
        return

    print(f"Gene: {args.gene}, targets: {len(gene_targets)} "
          f"(seqlet: {(gene_targets['mutation_type']=='seqlet_mut').sum()}, "
          f"negctrl: {(gene_targets['mutation_type']=='neg_control').sum()})")

    print("Loading model...")
    model, expr_df, expr_cols = load_model(study_name, study_full)
    model.module.cuda()

    gene_info = get_gene_info(study_name)
    if args.gene not in gene_info:
        print(f"Gene {args.gene} not found in BED file")
        return

    gene_entries = gene_info[args.gene]
    target_chroms = gene_targets["chromosome"].unique()
    selected_entry = next(
        (e for e in gene_entries if e["chrom"] in target_chroms),
        gene_entries[0])

    gene_idx = selected_entry["index"]
    tss_pos = selected_entry["tss"]

    ds = load_genome_dataset(study_name)
    seq_orig = ds[gene_idx]
    sequence_key = ds.get_interval_key(gene_idx)

    print("Computing WT prediction...")
    pred_wt = predict_single(model, seq_orig, sequence_key)
    print(f"  WT prediction shape: {pred_wt.shape}")

    win_start = tss_pos - HALF_CONTEXT

    n_targets = len(gene_targets)
    n_seeds = args.n_seeds
    pred_mt_all = np.zeros((n_targets, n_seeds, len(expr_cols)), dtype=np.float32)
    target_ids, mutation_types, site_ids = [], [], []
    mut_starts, mut_ends, chroms = [], [], []

    for ti, (_, row) in enumerate(gene_targets.iterrows()):
        mut_start_genomic = int(row["mut_start"])
        mut_end_genomic = int(row["mut_end"])
        mut_type = row["mutation_type"]

        rel_start = mut_start_genomic - win_start
        rel_end = mut_end_genomic - win_start

        if rel_start < 0 or rel_end > CONTEXT_LENGTH:
            print(f"  Skipping target {ti} (out of bounds)")
            pred_mt_all[ti, :, :] = pred_wt[np.newaxis, :]
        else:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 10000 + ti)
                seq_mut = shuffle_region(seq_orig, rel_start, rel_end, rng)
                pred_mt_all[ti, seed, :] = predict_single(model, seq_mut, sequence_key)

        target_ids.append(row["target_id"])
        mutation_types.append(mut_type)
        site_ids.append(row["site_id"])
        mut_starts.append(mut_start_genomic)
        mut_ends.append(mut_end_genomic)
        chroms.append(row["chromosome"])

        if (ti + 1) % 10 == 0 or ti == n_targets - 1:
            print(f"  Processed {ti + 1}/{n_targets} targets")

    out_path = os.path.join(outdir, f"{args.gene}_predictions.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("wt_pred", data=pred_wt)
        f.create_dataset("mt_pred", data=pred_mt_all)
        f.create_dataset("mt_pred_mean", data=pred_mt_all.mean(axis=1))

        meta = f.create_group("metadata")
        meta.create_dataset("gene", data=args.gene)
        meta.create_dataset("n_seeds", data=n_seeds)
        dt = h5py.string_dtype()
        meta.create_dataset("expr_cols", data=np.array(expr_cols, dtype=object), dtype=dt)
        meta.create_dataset("target_ids", data=np.array(target_ids, dtype=object), dtype=dt)
        meta.create_dataset("mutation_types", data=np.array(mutation_types, dtype=object), dtype=dt)
        meta.create_dataset("site_ids", data=np.array(site_ids, dtype=object), dtype=dt)
        meta.create_dataset("mut_starts", data=np.array(mut_starts, dtype=np.int64))
        meta.create_dataset("mut_ends", data=np.array(mut_ends, dtype=np.int64))
        meta.create_dataset("chromosomes", data=np.array(chroms, dtype=object), dtype=dt)
        meta.create_dataset("control_col", data=control_col)

    print(f"Saved: {out_path}")
    print(f"  WT: {pred_wt.shape}, MT: {pred_mt_all.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
