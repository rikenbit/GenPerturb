#!/usr/bin/env python3
"""
Fig. 4 / Fig. S12 attribution_analysis GimmeMotifs matrix aggregator.

This script is for the resume-safe attribution_analysis route used by the
lineage/master-regulator analyses, not for the paper Fig. 3e motif-recovery
panel. It converts per-perturbation
`attribution_analysis/gimme_results/{study}/{pert}/gimme.roc.report.txt` files
into `gimme_motif_pvalue_matrix.tsv`, which can be used by Fig. S12
`32_master_regulator_upset.py` for the GenPerturb(GimmeMotifs) evidence layer.

Do not confuse this with `scripts/attribution_evaluation/32_summary_gimmemotifs.py`
or `36_summary_combined_motif.py`, which generate the adopted Fig. 3e / Fig. S10a
paper motif summaries under `figures/{study}/gimmemotifs` and
`figures/{study}/combined_motif`.
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd


def parse_gimme_report(report_path: Path) -> pd.Series:
    """Parse gimme.roc.report.txt and return log10 P-value as a Series keyed by Motif."""
    df = pd.read_csv(report_path, sep="\t")
    return df.set_index("Motif")["log10 P-value"]


def aggregate_gimme_results(
    gimme_dir: Path,
    study_full: str,
    tasks_path: Path = None,
) -> pd.DataFrame:
    """
    Aggregate gimme.roc.report.txt across all perturbations.

    Parameters
    ----------
    gimme_dir : Path
        Directory containing per-perturbation gimme results.
    study_full : str
        Study full name (study__suffix).
    tasks_path : Path, optional
        Path to tasks.txt. If provided, use perturbation list from tasks.

    Returns
    -------
    pd.DataFrame
        Matrix with motifs as rows, perturbations as columns, log10 P-values as values.
    """
    study_gimme_dir = gimme_dir / study_full

    if not study_gimme_dir.exists():
        print(f"[ERROR] GimmeMotifs directory not found: {study_gimme_dir}")
        sys.exit(1)

    # Collect perturbation directories
    if tasks_path and tasks_path.exists():
        tasks = pd.read_csv(tasks_path, sep="\t", header=None,
                            names=["task_id", "study", "study_suffix", "model", "pert"])
        pert_list = [p.replace("/", "_") for p in tasks["pert"].tolist()]
    else:
        pert_list = sorted([
            d.name for d in study_gimme_dir.iterdir()
            if d.is_dir() and (d / "gimme.roc.report.txt").exists()
        ])

    results = {}
    for safe_pert in pert_list:
        report = study_gimme_dir / safe_pert / "gimme.roc.report.txt"
        if not report.exists():
            print(f"[WARN] No report for {safe_pert}, skipping")
            continue
        try:
            results[safe_pert] = parse_gimme_report(report)
        except Exception as e:
            print(f"[WARN] Failed to parse {report}: {e}")
            continue

    if not results:
        print("[ERROR] No GimmeMotifs results found")
        sys.exit(1)

    # Combine into matrix (motifs x perturbations), fill missing with 0
    matrix = pd.DataFrame(results).fillna(0)

    print(f"[INFO] Aggregated {matrix.shape[1]} perturbations, {matrix.shape[0]} motifs")
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate attribution_analysis GimmeMotifs results into the "
            "Fig. 4/Fig. S12 motif x perturbation matrix"
        )
    )
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--study-suffix", required=True, help="Study suffix")
    parser.add_argument(
        "--input-base", default="attribution_analysis",
        help="Base directory for GimmeMotifs output (default: attribution_analysis)"
    )
    parser.add_argument(
        "--output-base", default="attribution_analysis",
        help="Base directory for output (default: attribution_analysis)"
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Path to tasks.txt (optional, for perturbation list)"
    )
    args = parser.parse_args()

    study_full = f"{args.study}__{args.study_suffix}"
    gimme_dir = Path(args.input_base) / "gimme_results"
    tasks_path = Path(args.tasks) if args.tasks else Path(args.input_base) / "tasks.txt"

    matrix = aggregate_gimme_results(gimme_dir, study_full, tasks_path)

    # Save
    output_dir = Path(args.output_base) / "gimme_results" / study_full
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gimme_motif_pvalue_matrix.tsv"
    matrix.to_csv(output_path, sep="\t")

    print(f"[INFO] Saved: {output_path}")
    print(f"[INFO] Shape: {matrix.shape} (motifs x perturbations)")


if __name__ == "__main__":
    main()
