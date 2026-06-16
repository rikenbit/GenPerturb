#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_TAG = "alphagenome_transfer_epoch100_batch256_adamw5e3"
RAW_H5_SUFFIX = "_raw_attribution.h5"

STUDY_CONFIG = {
    "Martin": {
        "tag": "MartinRufino2025_mixscape_exnp_train",
        "ctrl_col": "MartinRufino.NT",
        "pert_prefix": "MartinRufino.",
    },
    "Norman": {
        "tag": "NormanWeissman2019_filtered_mixscape_exnp_train",
        "ctrl_col": "Norman.NT",
        "pert_prefix": "Norman.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=list(STUDY_CONFIG), default="Martin")
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Output/cache root. Default: "
            "figures/{study_tag}__{model_tag}/selected_martin_attribution_figures"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-genes", type=int, default=20)
    parser.add_argument(
        "--recompute-from-h5",
        action="store_true",
        help="Heavy mode: read attribution_pert H5 files and overwrite local per-pert TSVs.",
    )
    return parser.parse_args()


def default_work_dir(study: str, model_tag: str) -> Path:
    cfg = STUDY_CONFIG[study]
    return REPO_ROOT / "figures" / f"{cfg['tag']}__{model_tag}" / "selected_martin_attribution_figures"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def local_dirs(study: str, work_dir: Path) -> tuple[Path, Path, Path]:
    out_dir = work_dir / "axis_validation" / study
    tab_dir = out_dir / "tables"
    per_pert_dir = tab_dir / "per_pert"
    per_pert_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, tab_dir, per_pert_dir


def load_obs_and_fit(study: str, model_tag: str) -> tuple[pd.DataFrame, list[str]]:
    cfg = STUDY_CONFIG[study]
    train_path = REPO_ROOT / "data" / f"{cfg['tag']}.tsv"
    pred_path = REPO_ROOT / "prediction" / f"{cfg['tag']}__{model_tag}" / "prediction.npy"

    train = pd.read_csv(train_path, sep="\t")
    if "training" in train.columns:
        train = train.drop(columns=["training"])
    cond_cols = [c for c in train.columns if c != "Gene"]

    pred = np.load(pred_path)
    if pred.shape != (train.shape[0], len(cond_cols)):
        raise ValueError(
            f"prediction shape {pred.shape} does not match training matrix "
            f"({train.shape[0]}, {len(cond_cols)})"
        )

    fit = pd.DataFrame(pred, columns=cond_cols)
    fit.insert(0, "Gene", train["Gene"].values)
    obs = train.rename(columns={c: f"{c}__obs" for c in cond_cols})
    fit = fit.rename(columns={c: f"{c}__fit" for c in cond_cols})
    obs = obs.drop_duplicates(subset=["Gene"], keep="first").set_index("Gene")
    fit = fit.drop_duplicates(subset=["Gene"], keep="first").set_index("Gene")
    return pd.concat([obs, fit], axis=1), cond_cols


def process_one_h5(task: dict) -> dict:
    pert = task["pert"]
    h5_path = task["h5"]
    rows = []
    n_skipped = 0
    start = time.time()

    with h5py.File(h5_path, "r") as handle:
        for gene in handle.keys():
            group = handle[gene]
            if "ixg" not in group or "ixg_fc" not in group:
                n_skipped += 1
                continue
            ixg = group["ixg"][:]
            ixg_fc = group["ixg_fc"][:]
            ixg_ctrl = ixg - ixg_fc
            rows.append(
                {
                    "gene": gene,
                    "sum_abs_ctrl": float(np.abs(ixg_ctrl).sum()),
                    "sum_abs_pert": float(np.abs(ixg).sum()),
                    "sum_abs_diff": float(np.abs(ixg_fc).sum()),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["obs_ctrl"] = task["obs_ctrl"].reindex(df["gene"]).values
        df["obs_pert"] = task["obs_pert"].reindex(df["gene"]).values
        df["obs_abs_logFC"] = np.abs(df["obs_pert"] - df["obs_ctrl"])
        df["fit_ctrl"] = task["fit_ctrl"].reindex(df["gene"]).values
        df["fit_pert"] = task["fit_pert"].reindex(df["gene"]).values
        df["fit_abs_FC"] = np.abs(df["fit_pert"] - df["fit_ctrl"])
        df.insert(0, "perturbation", pert)

    return {
        "pert": pert,
        "df": df,
        "h5": str(h5_path),
        "elapsed": time.time() - start,
        "n_skipped": n_skipped,
    }


def recompute_from_h5(study: str, model_tag: str, per_pert_dir: Path, workers: int) -> list[str]:
    cfg = STUDY_CONFIG[study]
    expr, cond_cols = load_obs_and_fit(study, model_tag)
    print(f"[load] observed + fitted: {expr.shape[0]:,} genes x {len(cond_cols)} conditions")

    attr_dir = REPO_ROOT / "attribution_pert" / f"{cfg['tag']}__{model_tag}"
    if not attr_dir.exists():
        raise FileNotFoundError(f"attribution_pert directory not found: {attr_dir}")

    ctrl_col = cfg["ctrl_col"]
    tasks = []
    skipped = []
    for pert_dir in sorted(p for p in attr_dir.iterdir() if p.is_dir() and p.name.startswith(cfg["pert_prefix"])):
        pert = pert_dir.name
        if pert == ctrl_col:
            continue
        obs_col = f"{pert}__obs"
        fit_col = f"{pert}__fit"
        h5_path = pert_dir / f"{pert}{RAW_H5_SUFFIX}"
        if obs_col not in expr.columns:
            print(f"[skip] {pert}: not in expression matrix")
            continue
        if not h5_path.exists():
            print(f"[skip] {pert}: no all-gene H5 ({h5_path})")
            skipped.append(pert)
            continue
        tasks.append(
            {
                "pert": pert,
                "h5": h5_path,
                "obs_ctrl": expr[f"{ctrl_col}__obs"],
                "obs_pert": expr[obs_col],
                "fit_ctrl": expr[f"{ctrl_col}__fit"],
                "fit_pert": expr[fit_col],
            }
        )

    n_workers = max(1, min(workers, len(tasks)))
    print(f"[run] recomputing {len(tasks)} perturbations with {n_workers} workers")
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_h5, tasks), 1):
            df = result["df"]
            pert = result["pert"]
            if df.empty:
                print(f"[{i}/{len(tasks)}] {pert}: no rows")
                continue
            out = per_pert_dir / f"{pert}.tsv"
            df.to_csv(out, sep="\t", index=False)
            print(
                f"[{i}/{len(tasks)}] {pert}: {len(df):,} genes in "
                f"{result['elapsed']:.0f}s -> {display_path(out)}"
            )
    return skipped


def correlate(df: pd.DataFrame) -> list[dict]:
    attr_cols = {
        "sum_abs_ctrl": "Sum |Ctrl attribution|",
        "sum_abs_pert": "Sum |Pert attribution|",
        "sum_abs_diff": "Sum |Diff attribution|",
    }
    target_cols = {
        "obs_ctrl": ("observed control expression", "observed"),
        "obs_pert": ("observed perturbation expression", "observed"),
        "obs_abs_logFC": ("observed |log fold-change|", "observed"),
        "fit_ctrl": ("fitted control expression", "fitted"),
        "fit_pert": ("fitted perturbation expression", "fitted"),
        "fit_abs_FC": ("fitted |fold-change|", "fitted"),
    }
    rows = []
    for attr_col, attr_label in attr_cols.items():
        for target_col, (target_label, target_kind) in target_cols.items():
            sub = df[[attr_col, target_col]].dropna()
            if len(sub) < 5:
                pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(sub[attr_col], sub[target_col])
                spearman_rho, spearman_p = stats.spearmanr(sub[attr_col], sub[target_col])
            rows.append(
                {
                    "attribution": attr_label,
                    "attribution_key": attr_col,
                    "target": target_label,
                    "target_key": target_col,
                    "target_kind": target_kind,
                    "n": int(len(sub)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                }
            )
    return rows


def rebuild_summaries(
    study: str,
    model_tag: str,
    work_dir: Path,
    tab_dir: Path,
    per_pert_dir: Path,
    min_genes: int,
    skipped_no_h5: list[str],
) -> None:
    all_corrs = []
    pert_index = []
    tsvs = sorted(per_pert_dir.glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No local per-pert TSVs found under {per_pert_dir}")

    for tsv in tsvs:
        df = pd.read_csv(tsv, sep="\t")
        pert = str(df["perturbation"].iloc[0]) if "perturbation" in df.columns and len(df) else tsv.stem
        usable = df.dropna(subset=["obs_ctrl", "obs_pert", "fit_ctrl", "fit_pert"])
        used = len(usable) >= min_genes
        pert_index.append({"perturbation": pert, "n_genes": int(len(usable)), "local_tsv": str(tsv), "used": used})
        if not used:
            continue
        for row in correlate(usable):
            row["perturbation"] = pert
            row["n_genes"] = int(len(usable))
            all_corrs.append(row)

    pd.DataFrame(pert_index).to_csv(tab_dir / "perturbation_index.tsv", sep="\t", index=False)
    corr_df = pd.DataFrame(all_corrs)
    if not corr_df.empty:
        corr_df = corr_df[
            [
                "perturbation",
                "attribution",
                "attribution_key",
                "target",
                "target_key",
                "target_kind",
                "n",
                "n_genes",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        ]
    corr_df.to_csv(tab_dir / "per_pert_correlations.tsv", sep="\t", index=False)

    cfg = STUDY_CONFIG[study]
    config = {
        "study": study,
        "tag": cfg["tag"],
        "model_tag": model_tag,
        "ctrl_col": cfg["ctrl_col"],
        "min_genes": min_genes,
        "mode": "production cache under figures model directory",
        "work_dir": str(work_dir),
        "n_perturbations_used": int(sum(1 for row in pert_index if row["used"])),
        "n_perturbations_skipped_no_h5": len(skipped_no_h5),
        "perturbations_skipped_no_h5": skipped_no_h5,
    }
    (tab_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[saved] {display_path(tab_dir / 'per_pert_correlations.tsv')} ({len(corr_df):,} rows)")
    print(f"[saved] {display_path(tab_dir / 'perturbation_index.tsv')}")


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir if args.work_dir is not None else default_work_dir(args.study, args.model_tag)
    work_dir = work_dir.resolve()
    _, tab_dir, per_pert_dir = local_dirs(args.study, work_dir)
    print("=" * 72)
    print(f"  all-gene attribution-axis summaries - {args.study}")
    print("=" * 72)
    print(f"[work-dir] {display_path(work_dir)}")

    skipped_no_h5: list[str] = []
    if args.recompute_from_h5:
        skipped_no_h5 = recompute_from_h5(args.study, args.model_tag, per_pert_dir, args.workers)
    else:
        print("[mode] cache-only; not reading attribution_pert H5 files")

    rebuild_summaries(args.study, args.model_tag, work_dir, tab_dir, per_pert_dir, args.min_genes, skipped_no_h5)
    print("[done]")


if __name__ == "__main__":
    main()
