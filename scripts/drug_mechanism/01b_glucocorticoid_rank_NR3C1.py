#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _common import (  # noqa: E402
    PAPER_RCPARAMS,
    load_drug_mech_data,
)
from _glucocorticoid_pubchem import (  # noqa: E402
    PUBCHEM_GR_SCREEN_TSV,
    load_pubchem_gr_set,
)
from _rank_utils import rank_percentile_matrix  # noqa: E402

plt.rcParams.update(PAPER_RCPARAMS)

STUDY_FULL = "JialongJiang2024_CD8T_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
OUTDIR = ROOT / "figures" / STUDY_FULL / "drug_mechanism"

GR_COLOR    = "#C0392B"
OTHER_COLOR = "#9AA0A6"
MOTIF = "NR3C1"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    if p < 1e-4:
        return f"p = {p:.1e}"
    return f"p = {p:.3g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTDIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    print("[drug_mech.01b] loading data ...")
    gr_set     = load_pubchem_gr_set()
    print(f"[drug_mech.01b]   GR set input = {PUBCHEM_GR_SCREEN_TSV}")
    data = load_drug_mech_data()
    drug_names = data["drug_names"]

    rank_pct = rank_percentile_matrix(
        data["tfm_abs"], exclude_zero=False, method="min",
    )
    if MOTIF not in rank_pct.index:
        raise RuntimeError(f"{MOTIF} missing from rank_pct matrix")

    in_class = drug_names.isin(gr_set).values
    vals = rank_pct.loc[MOTIF].values.astype(float)
    audit = data["inclusion_audit"].copy()
    ranks = pd.Series(vals, index=drug_names.index)
    audit["rank"] = audit.perturbation.map(ranks)
    screen = pd.read_csv(PUBCHEM_GR_SCREEN_TSV, sep="\t").fillna("")
    for compound in sorted(set(screen.drug_name) - set(audit.compound)):
        audit.loc[len(audit)] = dict(perturbation="", compound=compound, included=0,
                                    exclusion_reason="absent_from_expression_and_motif_inputs", rank=np.nan)
    audit["group"] = np.where(audit.compound.isin(gr_set), "corticosteroid", "other")
    evidence = screen.groupby("drug_name", sort=False).apply(
        lambda d: " | ".join((d.pubchem_annotation + ": " + d.evidence_snippet).unique()), include_groups=False)
    audit["annotation_evidence"] = audit.compound.map(evidence).fillna("not present in frozen PubChem screen")
    audit.rename(columns={"compound": "compound_name"}, inplace=True)
    audit["compound"] = np.where(audit.perturbation.ne(""), audit.perturbation, "screen_only:" + audit.compound_name)
    audit["analysis_unit"] = "perturbation_condition"
    audit.to_csv(args.out / "compound_groups.tsv", sep="\t", index=False)
    collapsed = audit[audit.included.eq(1)].groupby(["compound_name", "group"], as_index=False).agg(
        rank=("rank", "median"), n_conditions=("perturbation", "size"), annotation_evidence=("annotation_evidence", "first"))
    collapsed.rename(columns={"compound_name": "compound"}, inplace=True)
    collapsed["included"], collapsed["exclusion_reason"] = 1, ""
    collapsed["analysis_unit"] = "unique_compound_median_across_conditions_sensitivity"
    collapsed.to_csv(args.out / "unique_compound_sensitivity.tsv", sep="\t", index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/additional_analysis/fig6a_compound_rank_test.py"),
                    "--input", str(args.out / "unique_compound_sensitivity.tsv"),
                    "--out", str(args.out / "compound_sensitivity")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/additional_analysis/fig6a_compound_rank_test.py"),
                    "--input", str(args.out / "compound_groups.tsv"),
                    "--out", str(args.out / "tie_aware")], check=True)

    nondetect = vals <= 1e-12
    n_total      = len(vals)
    n_nondet     = int(nondetect.sum())
    n_nondet_gr  = int((nondetect & in_class).sum())
    n_nondet_oth = int((nondetect & ~in_class).sum())

    print(f"[drug_mech.01b]   n compounds total = {n_total};  "
          f"non-detected (rank=0) = {n_nondet}  "
          f"(GR={n_nondet_gr}, other={n_nondet_oth})")

    other_mask = ~in_class
    tests = pd.read_csv(args.out / "tie_aware/nr3c1_tie_aware_tests.tsv", sep="\t")
    p_wilcox = float(tests.loc[tests.alternative.eq("two-sided"), "pvalue"].iloc[0])
    print(f"[drug_mech.01b]   Mann-Whitney U, two-sided {fmt_p(p_wilcox)}  "
          f"(GR n={in_class.sum()}, other n={other_mask.sum()})")

    rng = np.random.default_rng(42)
    x_oth = 0 + rng.uniform(-0.07, 0.07, size=int(other_mask.sum()))
    x_in  = 1 + rng.uniform(-0.07, 0.07, size=int(in_class.sum()))

    y_oth = vals[other_mask].copy()
    y_in  = vals[in_class].copy()
    oth_nd = nondetect[other_mask]
    in_nd  = nondetect[in_class]
    y_oth[oth_nd] += rng.uniform(-0.012, 0.012, size=int(oth_nd.sum()))
    y_in[in_nd]   += rng.uniform(-0.012, 0.012, size=int(in_nd.sum()))

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.scatter(x_oth, y_oth, s=42, color=OTHER_COLOR,
               alpha=0.65, edgecolor="gray", lw=0.3, zorder=2)
    ax.scatter(x_in,  y_in, s=130, color=GR_COLOR,
               alpha=0.95, edgecolor="black", lw=0.7, zorder=3)
    ax.plot([-0.32, 0.32], [np.median(vals[other_mask])] * 2,
            color="black", lw=2.0, zorder=4)
    ax.plot([0.68, 1.32], [np.median(vals[in_class])] * 2,
            color=GR_COLOR, lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Other conditions\n(n={other_mask.sum()})",
                        f"Corticosteroid conditions\n(n={in_class.sum()})"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("within-compound\nrank percentile (1 = top motif)")
    ax.set_title(MOTIF, pad=8, fontweight="bold")
    ax.grid(axis="y", linestyle=":", lw=0.4, alpha=0.4)

    txt = (f"{fmt_p(p_wilcox)}\n"
           f"Mann-Whitney U (two-sided)\n"
           f"rank: denom = {rank_pct.shape[0]} motifs\n"
           f"non-detected (rank=0): n = {n_nondet}\n"
           f"  (GR={n_nondet_gr}, other={n_nondet_oth})")
    ax.text(0.02, 0.98, txt,
            transform=ax.transAxes,
            fontsize=12, va="top", ha="left",
            family="DejaVu Sans Mono",
            bbox=dict(facecolor="white", edgecolor="#888",
                      alpha=0.85, boxstyle="round,pad=0.35"))

    fig.tight_layout()
    base = args.out / "01b_glucocorticoid_rank_NR3C1"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)
    print(f"[drug_mech.01b] wrote {base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
