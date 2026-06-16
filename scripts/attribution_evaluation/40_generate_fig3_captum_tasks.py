#!/usr/bin/env python3
"""
Generate explicit Fig. 3 Captum task lists.

This script intentionally separates the two Fig. 3 Captum inputs that used to
be easy to confuse:

  fig3d_gtgenes
    Martin enhancer AUPRC route. Computes attribution for Martin Table S3
    TF-sensitive genes with `10_captum.py union_genes --output-suffix _gtgenes`.

  fig3e_top200
    Motif-discovery route. Computes root-output top-200 variable-gene
    attribution with `10_captum.py variable_genes condition`.

Both routes write to the paper root `attribution/{study}__{suffix}/...`.
They are distinct from the Fig. 4 `attribution_analysis/` array route.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


CONDITION_PERTS = {
    "NormanWeissman2019": [
        "Norman.IRF1", "Norman.TP73", "Norman.CEBPA", "Norman.HNF4A",
        "Norman.FOXA1", "Norman.AHR", "Norman.PRDM1", "Norman.SPI1",
        "Norman.SNAI1", "Norman.KMT2A", "Norman.CEBPB", "Norman.JUN",
        "Norman.ETS2", "Norman.EGR1",
    ],
    "MartinRufino2025_mixscape_exnp": [
        "MartinRufino.BCL11A", "MartinRufino.FOSL1", "MartinRufino.GATA1",
        "MartinRufino.GATA2", "MartinRufino.GFI1B", "MartinRufino.KLF1",
        "MartinRufino.LDB1", "MartinRufino.LMO2", "MartinRufino.MYB",
        "MartinRufino.NFE2", "MartinRufino.RUNX1", "MartinRufino.SPI1",
        "MartinRufino.TAL1",
    ],
}


def default_study_suffix(model: str) -> str:
    return f"{model}_transfer_epoch100_batch256_adamw5e3"


def data_tsv_path(study: str, model: str) -> Path:
    if model == "enformer":
        return Path(f"data/{study}_enformer.tsv")
    if model.startswith("alphagenome_fold_"):
        return Path(f"data/{study}_{model}.tsv")
    return Path(f"data/{study}.tsv")


def bed_path(study: str, model: str) -> Path:
    if model.startswith("alphagenome_fold_"):
        return Path(f"fasta/{study}_{model}.bed")
    return Path(f"fasta/{study}.bed")


def load_perturbations_from_tsv(study: str, model: str) -> list[str]:
    df = pd.read_csv(data_tsv_path(study, model), sep="\t", index_col=0, nrows=1)
    value_cols = list(df.columns[1:])
    return value_cols[1:]


def select_perturbations(study: str, model: str, target: str, perts_arg: str | None) -> list[str]:
    if perts_arg:
        selected = [p.strip() for p in perts_arg.split(",") if p.strip()]
    elif target == "condition":
        selected = None
        for key, perts in CONDITION_PERTS.items():
            if key in study:
                selected = list(perts)
                break
        if selected is None:
            selected = load_perturbations_from_tsv(study, model)
    elif target == "all":
        selected = load_perturbations_from_tsv(study, model)
    else:
        raise ValueError(
            f"Unsupported task-generation target '{target}'. "
            "Use condition/all or provide --perts."
        )

    tsv_path = data_tsv_path(study, model)
    if tsv_path.exists():
        available_set = set(load_perturbations_from_tsv(study, model))
        missing = [p for p in selected if p not in available_set]
        if missing:
            raise ValueError(f"Selected perturbations are absent from {tsv_path}: {missing}")
    else:
        print(f"[WARN] {tsv_path} not found; skipping perturbation-column validation.")
    return selected


def read_base_genes(path: Path) -> set[str]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "gene", "score", "strand", "split"],
        comment="#",
    )
    return set(df["gene"].dropna().astype(str))


def load_martin_table_s3_genes(xlsx_path: Path) -> dict[str, set[str]]:
    df = pd.read_excel(xlsx_path, sheet_name="TF_sensitive_genes", header=2)
    df = df.dropna(subset=["gene_ID", "perturbation_name"])
    grouped = df.groupby("perturbation_name")["gene_ID"]
    return {str(tf): set(s.astype(str)) for tf, s in grouped}


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""))


def generate_fig3d_gtgenes(args: argparse.Namespace) -> None:
    suffix = args.suffix or default_study_suffix(args.model)
    study_full = f"{args.study}__{suffix}"
    selected_perts = select_perturbations(args.study, args.model, args.target, args.perts)
    base_genes = read_base_genes(bed_path(args.study, args.model))
    gtgenes_by_tf = load_martin_table_s3_genes(Path(args.gtgenes_xlsx))

    task_rows: list[str] = []
    summary_rows: list[dict[str, object]] = []

    for task_id, pert in enumerate(selected_perts):
        tf = pert.split(".")[-1]
        if tf not in gtgenes_by_tf:
            raise ValueError(f"{tf} from {pert} is absent from Martin Table S3")
        gtgenes = gtgenes_by_tf[tf]
        keep = sorted(gtgenes & base_genes)
        if not keep:
            raise ValueError(f"No Table S3 genes overlap {bed_path(args.study, args.model)} for {pert}")

        safe_pert = pert.replace("/", "_")
        gene_list = Path("attribution") / study_full / safe_pert / f"{safe_pert}_gtgenes_genes.txt"
        write_lines(gene_list, keep)

        task_rows.append(
            "\t".join(
                [
                    str(task_id),
                    "fig3d_gtgenes",
                    args.study,
                    suffix,
                    args.model,
                    args.target,
                    pert,
                    str(gene_list),
                    args.output_suffix,
                ]
            )
        )
        summary_rows.append(
            {
                "task_id": task_id,
                "pert": pert,
                "tf": tf,
                "n_table_s3_genes": len(gtgenes),
                "n_genes_in_model_bed": len(keep),
                "gene_list": str(gene_list),
            }
        )

    write_lines(Path(args.output), task_rows)
    summary_path = Path("attribution") / study_full / "_fig3d_gtgenes_captum_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    print(f"Generated {len(task_rows)} Fig. 3d gtgenes tasks -> {args.output}")
    print(f"Summary -> {summary_path}")
    print(f"Study: {study_full}")
    print(f"Target: {args.target}; output suffix: {args.output_suffix}")


def generate_fig3e_top200(args: argparse.Namespace) -> None:
    suffix = args.suffix or default_study_suffix(args.model)
    selected_perts = select_perturbations(args.study, args.model, args.target, args.perts)

    task_rows = [
        "\t".join(
            [
                str(task_id),
                "fig3e_top200",
                args.study,
                suffix,
                args.model,
                args.mode,
                args.target,
                pert,
            ]
        )
        for task_id, pert in enumerate(selected_perts)
    ]
    write_lines(Path(args.output), task_rows)

    print(f"Generated {len(task_rows)} Fig. 3e top-200 tasks -> {args.output}")
    print(f"Study: {args.study}__{suffix}")
    print(f"Mode/target: {args.mode}/{args.target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 3 Captum task lists")
    sub = parser.add_subparsers(dest="route", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("study")
    common.add_argument("model")
    common.add_argument("--suffix", default=None)
    common.add_argument("--target", default="condition")
    common.add_argument("--perts", default=None, help="Comma-separated perturbation override")
    common.add_argument("--output", required=True)

    p3d = sub.add_parser("fig3d_gtgenes", parents=[common])
    p3d.add_argument(
        "--gtgenes-xlsx",
        default="data/science.ads7951_tables_s1_to_s6/science.ads7951_table_s3.xlsx",
    )
    p3d.add_argument("--output-suffix", default="_gtgenes")
    p3d.set_defaults(func=generate_fig3d_gtgenes)

    p3e = sub.add_parser("fig3e_top200", parents=[common])
    p3e.add_argument("--mode", default="variable_genes")
    p3e.set_defaults(func=generate_fig3e_top200)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
