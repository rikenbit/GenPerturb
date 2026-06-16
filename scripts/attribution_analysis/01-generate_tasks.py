#!/usr/bin/env python3
import sys
import argparse
import pandas as pd


def get_tsv_path(study: str, model: str) -> str:
    """Determine TSV path matching 10_captum.py ModelConfig logic."""
    if model == "enformer":
        return f"data/{study}_enformer.tsv"
    elif model.startswith("alphagenome_fold_"):
        return f"data/{study}_{model}.tsv"
    return f"data/{study}.tsv"


def get_study_suffix(model: str, suffix_override: str = None) -> str:
    """Determine study suffix matching the project convention."""
    if suffix_override:
        return suffix_override
    # Default suffix for transfer learning (matches 00_sbatch.sh convention)
    return f"{model}_transfer_epoch100_batch256_adamw5e3"


def main():
    parser = argparse.ArgumentParser(
        description="Generate tasks.txt for parallel captum + peak-call jobs"
    )
    parser.add_argument("study", help="Study name (e.g. JialongJiang2024_CD8T_train)")
    parser.add_argument("model", help="Model name (e.g. alphagenome)")
    parser.add_argument(
        "--suffix", default=None,
        help="Study suffix override. Default: {model}_transfer_epoch100_batch256_adamw5e3"
    )
    parser.add_argument(
        "--output", default="attribution_analysis/tasks.txt",
        help="Output file path (default: attribution_analysis/tasks.txt)"
    )
    args = parser.parse_args()

    study = args.study
    model = args.model
    study_suffix = get_study_suffix(model, args.suffix)
    tsv_path = get_tsv_path(study, model)

    # Read TSV (same logic as 10_captum.py load_predictions)
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)

    # df.columns[0] = metadata (e.g. "training")
    # df.columns[1:] = value columns; first value column = control
    value_cols = df.columns[1:]
    ctrl_col = value_cols[0]
    perturbations = value_cols[1:].tolist()

    # Write tasks.txt
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as f:
        for i, pert in enumerate(perturbations):
            f.write(f"{i}\t{study}\t{study_suffix}\t{model}\t{pert}\n")

    print(f"Generated {len(perturbations)} tasks -> {args.output}")
    print(f"  Study:    {study}")
    print(f"  Suffix:   {study_suffix}")
    print(f"  Model:    {model}")
    print(f"  TSV:      {tsv_path}")
    print(f"  Control:  {ctrl_col}")
    print(f"  Perts:    {len(perturbations)} (first: {perturbations[0]}, last: {perturbations[-1]})")


if __name__ == "__main__":
    main()
