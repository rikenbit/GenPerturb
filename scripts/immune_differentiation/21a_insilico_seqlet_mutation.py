#!/usr/bin/env python
import argparse
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, CWD)
os.chdir(CWD)

from genperturb.model._genperturb import GenPerturb
from genperturb.dataloaders._genome import (
    GenomeIntervalDataset,
)
from genperturb.dataloaders._alphagenome_sequence import alphagenome_indices_to_one_hot

STUDY_NAME = "NormanWeissman2019_filtered_mixscape_exnp_train"
STUDY_FULL = "NormanWeissman2019_filtered_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
CONTROL_COL = "Norman.NT"
CONTEXT_LENGTH = 1_048_576
HALF_CONTEXT = CONTEXT_LENGTH // 2
FASTA = "fasta/GRCh38.p14.genome.fa"
SEQLET_FILE = "attribution_analysis/captum_union/seqlet_loci_all_core.tsv"
JASPAR_CLUSTERS = "reference/jaspar/clusters.tab"

DEFAULT_OUTDIR = f"figures/{STUDY_FULL}/seqlet_mutation/mutation_predictions"

MARKER_GENES = ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "GYPA",
                "ITGAM", "CSF3R", "LST1"]

CHAIN_TFS = {
    "Erythroid": ["KLF1", "GATA1", "TAL1"],
    "Granulocyte": ["SPI1", "IRF1", "IRF2", "IRF7",
                    "CEBPA", "CEBPB", "CEBPD", "CEBPE", "PRDM1"],
}

ERYTHROID_PERTS = [
    "Norman.CBL_CNN1", "Norman.CBL_PTPN12", "Norman.CBL_PTPN9",
    "Norman.CBL_UBASH3B", "Norman.SAMD1_PTPN12", "Norman.SAMD1_UBASH3B",
    "Norman.UBASH3B_CNN1", "Norman.UBASH3B_PTPN12", "Norman.UBASH3B_PTPN9",
    "Norman.UBASH3B_UBASH3A", "Norman.UBASH3B_ZBTB25", "Norman.BPGM_SAMD1",
    "Norman.PTPN1", "Norman.PTPN12_PTPN9", "Norman.PTPN12_UBASH3A",
    "Norman.PTPN12_ZBTB25",
]
GRANULOCYTE_PERTS = [
    "Norman.SPI1", "Norman.CEBPA", "Norman.CEBPB",
    "Norman.CEBPE_CEBPA", "Norman.CEBPE_RUNX1T1", "Norman.CEBPE_SPI1",
    "Norman.CEBPE", "Norman.ETS2_CEBPE", "Norman.KLF1_CEBPA",
    "Norman.FOSB_CEBPE",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", type=str, default=None,
                   help="Target gene symbol (one of the 9 marker genes)")
    p.add_argument("--all", action="store_true",
                   help="Process all 9 marker genes")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of random shuffle seeds per seqlet")
    p.add_argument("--max-seqlets", type=int, default=50,
                   help="Max seqlets per gene to use")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: figures/{STUDY_FULL}/seqlet_mutation/mutation_predictions)")
    return p.parse_args()


def build_chain_cluster_members():
    all_chain_tfs = set()
    for tfs in CHAIN_TFS.values():
        all_chain_tfs.update(tfs)

    clusters_df = pd.read_csv(JASPAR_CLUSTERS, sep="\t")
    cluster_members = set()

    for _, row in clusters_df.iterrows():
        names = set(row["name"].split(","))
        # Check if any chain TF is in this cluster
        if names & all_chain_tfs:
            cluster_members.update(names)

    return cluster_members


def select_chain_seqlets(gene, cluster_members):
    df = pd.read_csv(SEQLET_FILE, sep="\t")
    df = df[(df["study"] == "Norman") & (df["gene"] == gene)]

    mask = df["best_motif_gene"].isin(cluster_members)
    selected = df[mask].copy()

    selected = selected[selected["core_genomic_start"].notna()].copy()

    selected = selected.drop_duplicates(
        subset=["chromosome", "core_genomic_start", "core_genomic_end"]
    ).reset_index(drop=True)

    return selected


def load_model():
    df = pd.read_csv(f"data/{STUDY_NAME}.tsv", sep="\t", index_col=[0])
    hdf5 = f"data/{STUDY_NAME}_alphagenome.h5"

    model = GenPerturb(
        df,
        hdf5=hdf5,
        context_length=CONTEXT_LENGTH,
        pretrained="alphagenome",
        training_method="transfer",
        study=STUDY_FULL,
        emb_method="tss",
    )
    model.load_model()
    model.load_pretrained_model()
    model.training_method = "prediction"
    model.module.training_method = "prediction"
    model.module.eval()
    model.module.pretrained_model.eval()

    return model, df


def load_genome_dataset():
    bed = f"fasta/{STUDY_NAME}.bed"
    ds = GenomeIntervalDataset(
        bed_file=bed,
        fasta_file=FASTA,
        return_seq_indices=True,
        context_length=CONTEXT_LENGTH,
    )
    return ds


def get_gene_index(gene):
    bed = f"fasta/{STUDY_NAME}.bed"
    bed_df = pd.read_csv(bed, sep="\t", header=None,
                         names=["chr", "start", "end", "gene", "score",
                                "strand", "split"])
    idx = bed_df.index[bed_df["gene"] == gene].tolist()
    if not idx:
        return None
    return idx[0]


def shuffle_sequence(seq_indices, start, end, rng):
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


def genomic_to_window(tss_pos, genomic_start, genomic_end):
    win_start = tss_pos - HALF_CONTEXT
    rel_start = genomic_start - win_start
    rel_end = genomic_end - win_start
    return int(rel_start), int(rel_end)


def process_gene(gene, model, expr_df, ds, cluster_members,
                 n_seeds, max_seqlets, outdir):
    seqlets = select_chain_seqlets(gene, cluster_members)
    if len(seqlets) == 0:
        print(f"No chain TF seqlets found for {gene}")
        pd.DataFrame().to_csv(f"{outdir}/{gene}_mutation_predictions.tsv",
                              sep="\t", index=False)
        return

    if len(seqlets) > max_seqlets:
        if "best_qval" in seqlets.columns:
            seqlets_with_qval = seqlets[seqlets["best_qval"].notna()]
            seqlets_without_qval = seqlets[seqlets["best_qval"].isna()]
            seqlets_with_qval = seqlets_with_qval.sort_values("best_qval")
            seqlets = pd.concat([
                seqlets_with_qval.head(max_seqlets),
                seqlets_without_qval
            ]).head(max_seqlets)
        else:
            seqlets = seqlets.head(max_seqlets)

    print(f"\n{'='*60}")
    print(f"Gene: {gene} — {len(seqlets)} chain TF seqlets")
    print(f"{'='*60}")

    gene_idx = get_gene_index(gene)
    if gene_idx is None:
        print(f"  Gene {gene} not found in BED file")
        return

    bed_df = pd.read_csv(f"fasta/{STUDY_NAME}.bed", sep="\t", header=None,
                         names=["chr", "start", "end", "gene", "score",
                                "strand", "split"])
    gene_row = bed_df.iloc[gene_idx]
    tss_pos = gene_row["start"]

    all_pred_cols = [c for c in expr_df.columns if c not in ["Gene", "training"]]
    control_idx = all_pred_cols.index(CONTROL_COL)
    pert_cols = [c for c in all_pred_cols if c != CONTROL_COL]

    print(f"  Computing wildtype prediction (idx={gene_idx})...")
    seq_orig = ds[gene_idx]
    sequence_key = ds.get_interval_key(gene_idx)
    pred_wt_all = predict_single(model, seq_orig, sequence_key)
    pred_wt_control = float(pred_wt_all[control_idx])
    pert_indices = [i for i, c in enumerate(all_pred_cols) if c != CONTROL_COL]
    pred_wt = pred_wt_all[pert_indices]

    results = []
    for si, (_, row) in enumerate(seqlets.iterrows()):
        chrom = row["chromosome"]
        core_start = int(row["core_genomic_start"])
        core_end = int(row["core_genomic_end"])
        seqlet_start = int(row["genomic_start"])
        seqlet_end = int(row["genomic_end"])
        motif_gene = row.get("best_motif_gene", "")
        motif_match = row.get("best_match", "")
        is_revcomp = row.get("is_revcomp", False)
        core_length = int(row.get("core_length", core_end - core_start))

        rel_start, rel_end = genomic_to_window(tss_pos, core_start, core_end)

        if rel_start < 0 or rel_end > CONTEXT_LENGTH:
            print(f"  Skipping seqlet at {chrom}:{core_start}-{core_end} (out of bounds)")
            continue

        motif_len = rel_end - rel_start

        for seed in range(n_seeds):
            rng = np.random.RandomState(seed * 10000 + si)
            seq_mut = shuffle_sequence(seq_orig, rel_start, rel_end, rng)
            pred_mut_all = predict_single(model, seq_mut, sequence_key)
            pred_mut_control = float(pred_mut_all[control_idx])
            pred_mut = pred_mut_all[pert_indices]

            delta = pred_mut - pred_wt

            for pi, pc in enumerate(pert_cols):
                if pc in ERYTHROID_PERTS:
                    pert_lineage = "Erythroid"
                elif pc in GRANULOCYTE_PERTS:
                    pert_lineage = "Granulocyte"
                else:
                    pert_lineage = "Other"

                wt_fc = float(pred_wt[pi]) - pred_wt_control
                mt_fc = float(pred_mut[pi]) - pred_mut_control

                results.append({
                    "gene": gene,
                    "chromosome": chrom,
                    "seqlet_start": seqlet_start,
                    "seqlet_end": seqlet_end,
                    "core_genomic_start": core_start,
                    "core_genomic_end": core_end,
                    "motif_gene": motif_gene,
                    "motif_match": motif_match,
                    "is_revcomp": is_revcomp,
                    "core_length": core_length,
                    "motif_length": motif_len,
                    "seed": seed,
                    "target_perturbation": pc,
                    "pert_lineage": pert_lineage,
                    "pred_wt": float(pred_wt[pi]),
                    "pred_wt_control": pred_wt_control,
                    "wt_fc": wt_fc,
                    "pred_mut": float(pred_mut[pi]),
                    "pred_mut_control": pred_mut_control,
                    "mt_fc": mt_fc,
                    "delta": float(delta[pi]),
                })

        print(f"  [{si+1}/{len(seqlets)}] {chrom}:{core_start}-{core_end} "
              f"motif={motif_gene} core_len={motif_len}")

    result_df = pd.DataFrame(results)
    out_path = f"{outdir}/{gene}_mutation_predictions.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved {len(result_df)} rows to {out_path}")

    if len(result_df) > 0:
        summary = (
            result_df
            .groupby(["gene", "chromosome", "core_genomic_start", "core_genomic_end",
                       "motif_gene", "motif_match",
                       "target_perturbation", "pert_lineage"])
            .agg(
                pred_wt=("pred_wt", "first"),
                pred_mut_mean=("pred_mut", "mean"),
                delta_mean=("delta", "mean"),
                delta_std=("delta", "std"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        sum_path = f"{outdir}/{gene}_mutation_summary.tsv"
        summary.to_csv(sum_path, sep="\t", index=False)
        print(f"  Saved summary ({len(summary)} rows) to {sum_path}")


def main():
    args = parse_args()

    if args.gene is None and not args.all:
        print("Specify --gene GENE or --all")
        sys.exit(1)

    genes = MARKER_GENES if args.all else [args.gene]

    if args.outdir is None:
        outdir = DEFAULT_OUTDIR
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    cluster_members = build_chain_cluster_members()
    print(f"Chain TF cluster members: {len(cluster_members)} TFs")

    print("\nLoading model...")
    model, expr_df = load_model()
    model.module.cuda()

    ds = load_genome_dataset()

    for gene in genes:
        if gene not in MARKER_GENES:
            print(f"Warning: {gene} is not a marker gene, skipping")
            continue
        process_gene(gene, model, expr_df, ds,
                     cluster_members,
                     args.n_seeds, args.max_seqlets, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
