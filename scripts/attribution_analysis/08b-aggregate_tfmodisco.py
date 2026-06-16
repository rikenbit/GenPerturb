#!/usr/bin/env python3
"""
Fig. 4 / Fig. S11 / Fig. S12 attribution_analysis TF-MoDISco matrix aggregator.

This script is for the resume-safe attribution_analysis route used by the
lineage and master-regulator analyses, not for the paper Fig. 3e motif-recovery
panel. It converts per-perturbation
`attribution_analysis/tfmodisco/{study}/{pert}/modisco_result/*_MA_list.txt`
files into `tfmodisco_motif_matrix_{pos,neg,signed}.tsv`.

The pos/neg matrices are direct inputs to
`scripts/immune_differentiation/11_signature_axis_plots.py` (Fig. 4c / Fig. S11b)
and `scripts/immune_differentiation/32_master_regulator_upset.py` (Fig. S12).

Do not confuse this with `scripts/attribution_evaluation/34_summary_tfmodisco.py`
or `36_summary_combined_motif.py`, which generate the adopted Fig. 3e / Fig. S10a
paper motif summaries under `figures/{study}/tfmodisco` and
`figures/{study}/combined_motif`.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract_gene_from_match(match_str: str) -> str:
    """Extract gene name from JASPAR match string like 'MA1558.1_MA1558.1.SNAI1'."""
    parts = match_str.split(".")
    if len(parts) >= 3:
        return parts[-1]
    return match_str


def parse_ma_list(report_path: Path, qval_threshold: float = 1.0) -> dict:
    """
    Parse _MA_list.txt and return best q-values per motif gene, split by direction.

    Returns
    -------
    dict with keys 'pos' and 'neg', each a pd.Series keyed by motif gene name
    with values = best (min) q-value for that direction.
    """
    df = pd.read_csv(report_path, sep="\t")
    if df.empty:
        return {"pos": pd.Series(dtype=float), "neg": pd.Series(dtype=float)}

    # Extract gene name from match column
    df["motif_gene"] = df["match"].apply(extract_gene_from_match)

    results = {}
    for direction, prefix in [("pos", "pos_patterns"), ("neg", "neg_patterns")]:
        sub = df[df["pattern"].str.startswith(prefix)].copy()
        if sub.empty:
            results[direction] = pd.Series(dtype=float)
            continue
        # Best (min) q-value per motif gene
        best = sub.groupby("motif_gene")["qval"].min()
        # Apply threshold
        best = best[best <= qval_threshold]
        # Convert to -log10(qval), clamp small values
        best = -np.log10(best.clip(lower=1e-300))
        results[direction] = best

    return results


def aggregate_tfmodisco_results(
    tfmodisco_dir: Path,
    study_full: str,
    tasks_path: Path = None,
    qval_threshold: float = 1.0,
) -> dict:
    """
    Aggregate _MA_list.txt across all perturbations.

    Returns
    -------
    dict with keys 'pos', 'neg', 'signed', each a pd.DataFrame
    (motifs x perturbations).
    """
    study_dir = tfmodisco_dir / study_full

    if not study_dir.exists():
        print(f"[ERROR] TF-MoDISco directory not found: {study_dir}")
        sys.exit(1)

    # Collect perturbation directories
    if tasks_path and tasks_path.exists():
        tasks = pd.read_csv(tasks_path, sep="\t", header=None,
                            names=["task_id", "study", "study_suffix", "model", "pert"])
        pert_list = [p.replace("/", "_") for p in tasks["pert"].tolist()]
    else:
        pert_list = sorted([
            d.name for d in study_dir.iterdir()
            if d.is_dir() and (d / "modisco_result").is_dir()
        ])

    pos_results = {}
    neg_results = {}

    for safe_pert in pert_list:
        # Find _MA_list.txt
        report_dir = study_dir / safe_pert / "modisco_result"
        ma_files = list(report_dir.glob("*_MA_list.txt")) if report_dir.exists() else []
        if not ma_files:
            continue

        report = ma_files[0]
        try:
            parsed = parse_ma_list(report, qval_threshold=qval_threshold)
            if not parsed["pos"].empty:
                pos_results[safe_pert] = parsed["pos"]
            if not parsed["neg"].empty:
                neg_results[safe_pert] = parsed["neg"]
        except Exception as e:
            print(f"[WARN] Failed to parse {report}: {e}")
            continue

    if not pos_results and not neg_results:
        print("[ERROR] No TF-MoDISco results found")
        sys.exit(1)

    # Build matrices (motifs x perturbations), fill missing with 0
    pos_matrix = pd.DataFrame(pos_results).fillna(0) if pos_results else pd.DataFrame()
    neg_matrix = pd.DataFrame(neg_results).fillna(0) if neg_results else pd.DataFrame()

    # Signed matrix: union of all motifs across pos and neg
    all_motifs = sorted(set(
        list(pos_matrix.index if not pos_matrix.empty else []) +
        list(neg_matrix.index if not neg_matrix.empty else [])
    ))
    all_perts = sorted(set(
        list(pos_matrix.columns if not pos_matrix.empty else []) +
        list(neg_matrix.columns if not neg_matrix.empty else [])
    ))

    signed_matrix = pd.DataFrame(0.0, index=all_motifs, columns=all_perts)
    if not pos_matrix.empty:
        for m in pos_matrix.index:
            for p in pos_matrix.columns:
                signed_matrix.loc[m, p] += pos_matrix.loc[m, p]
    if not neg_matrix.empty:
        for m in neg_matrix.index:
            for p in neg_matrix.columns:
                signed_matrix.loc[m, p] -= neg_matrix.loc[m, p]

    n_pos = pos_matrix.shape[1] if not pos_matrix.empty else 0
    n_neg = neg_matrix.shape[1] if not neg_matrix.empty else 0
    print(f"[INFO] Aggregated: pos={n_pos} perts ({pos_matrix.shape[0] if not pos_matrix.empty else 0} motifs), "
          f"neg={n_neg} perts ({neg_matrix.shape[0] if not neg_matrix.empty else 0} motifs), "
          f"signed={signed_matrix.shape[1]} perts ({signed_matrix.shape[0]} motifs)")

    return {"pos": pos_matrix, "neg": neg_matrix, "signed": signed_matrix}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate attribution_analysis TF-MoDISco results into the "
            "Fig. 4/Fig. S11/Fig. S12 motif x perturbation matrices"
        )
    )
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--study-suffix", required=True, help="Study suffix")
    parser.add_argument(
        "--input-base", default="attribution_analysis",
        help="Base directory for TF-MoDISco output (default: attribution_analysis)"
    )
    parser.add_argument(
        "--output-base", default="attribution_analysis",
        help="Base directory for output (default: attribution_analysis)"
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Path to tasks.txt (optional, for perturbation list)"
    )
    parser.add_argument(
        "--qval-threshold", type=float, default=1.0,
        help="Q-value threshold for including motifs (default: 1.0, include all)"
    )
    args = parser.parse_args()

    study_full = f"{args.study}__{args.study_suffix}"
    tfmodisco_dir = Path(args.input_base) / "tfmodisco"
    tasks_path = Path(args.tasks) if args.tasks else None
    if tasks_path is None:
        # Try study-specific tasks file
        candidate = Path(args.input_base) / f"tasks_{study_full}.txt"
        if candidate.exists():
            tasks_path = candidate

    matrices = aggregate_tfmodisco_results(
        tfmodisco_dir, study_full, tasks_path,
        qval_threshold=args.qval_threshold,
    )

    # Save
    output_dir = Path(args.output_base) / "tfmodisco" / study_full
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["pos", "neg", "signed"]:
        mat = matrices[key]
        if mat.empty:
            print(f"[WARN] {key} matrix is empty, skipping")
            continue
        output_path = output_dir / f"tfmodisco_motif_matrix_{key}.tsv"
        mat.to_csv(output_path, sep="\t")
        print(f"[INFO] Saved: {output_path}  shape={mat.shape}")

    print("[DONE]")


if __name__ == "__main__":
    main()
