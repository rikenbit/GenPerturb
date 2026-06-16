#!/usr/bin/env python3
from __future__ import annotations

import sys
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
OUTDIR.mkdir(parents=True, exist_ok=True)

MOTIF       = "NR3C1"
TF_SYMBOLS  = ["NR3C1"]
GR_COLOR    = "#C0392B"
OTHER_COLOR = "#BDC3C7"


def family_signed_log2fc(fc_df: pd.DataFrame, tf_symbols: list[str]) -> pd.Series:
    avail = [g for g in tf_symbols if g in fc_df.columns]
    if not avail:
        return pd.Series(np.nan, index=fc_df.index)
    sub = fc_df[avail]
    idx_max = sub.abs().idxmax(axis=1)
    return pd.Series(
        [sub.loc[p, idx_max.loc[p]] if isinstance(idx_max.loc[p], str)
         else np.nan
         for p in sub.index],
        index=sub.index,
    )


def main():
    print("[drug_mech.02] loading data ...")
    gr_set     = load_pubchem_gr_set()
    print(f"[drug_mech.02]   GR set input = {PUBCHEM_GR_SCREEN_TSV}")
    data = load_drug_mech_data()
    fc_df      = data["fc_pseudo_df"]
    drug_names = data["drug_names"]

    # Constant 791-motif denominator (matches 01b). rankdata(method="min")
    # puts the zero-tie group at rank_pct = 0; detected motifs occupy the
    # upper portion of [0, 1].
    rank_pct = rank_percentile_matrix(
        data["tfm_abs"], exclude_zero=False, method="min",
    )

    if MOTIF not in rank_pct.index:
        raise RuntimeError(f"{MOTIF} missing from rank_pct matrix")

    perts = list(rank_pct.columns)
    in_gr = np.array([drug_names[p] in gr_set for p in perts], dtype=bool)
    print(f"[drug_mech.02]   compounds: {len(perts)};  GR: {in_gr.sum()}")

    x = family_signed_log2fc(fc_df, TF_SYMBOLS).reindex(perts).values
    y = rank_pct.loc[MOTIF].reindex(perts).values.astype(float)
    # method="min" puts non-detected (score=0) compounds at rank_pct = 0.
    nondetect = y <= 1e-12

    finite_x = np.isfinite(x)
    n_nondet     = int((nondetect & finite_x).sum())
    n_nondet_gr  = int((nondetect & in_gr & finite_x).sum())
    n_nondet_oth = int((nondetect & ~in_gr & finite_x).sum())

    # Slight vertical jitter for the y=0 tie cluster so dots are visible.
    rng = np.random.default_rng(42)
    y_plot = y.copy()
    y_plot[nondetect] += rng.uniform(-0.012, 0.012,
                                       size=int(nondetect.sum()))

    print(f"[drug_mech.02]   n = {finite_x.sum()};  non-detected pinned to 0: "
          f"{n_nondet}  GR={n_nondet_gr}, other={n_nondet_oth}")

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    other_plot = (~in_gr) & finite_x
    gr_plot    = in_gr & finite_x
    ax.scatter(x[other_plot], y_plot[other_plot], s=35, color=OTHER_COLOR,
               alpha=0.55, edgecolor="gray", lw=0.2, zorder=2,
               label=f"other (n={other_plot.sum()})")
    if gr_plot.any():
        ax.scatter(x[gr_plot], y_plot[gr_plot], s=110, color=GR_COLOR,
                   alpha=0.95, edgecolor="black", lw=0.7, zorder=4,
                   label=f"Corticosteroid (n={gr_plot.sum()})")

    ax.axvline(0, color="black", lw=0.4, ls=":")
    ax.set_xlabel(f"TF mRNA log2 FC  ({','.join(TF_SYMBOLS)})")
    ax.set_ylabel("motif within-compound\nrank percentile (1 = top motif)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{MOTIF} (within-compound rank percentile)", pad=8)
    ax.grid(linestyle=":", lw=0.4, alpha=0.4)

    txt = (f"n = {finite_x.sum()}\n"
           f"rank: denom = 791 motifs\n"
           f"non-detected (rank=0): n = {n_nondet}\n"
           f"  (GR={n_nondet_gr}, other={n_nondet_oth})")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=12, va="top", ha="left",
            family="DejaVu Sans Mono",
            bbox=dict(facecolor="white", edgecolor="#888",
                      alpha=0.85, boxstyle="round,pad=0.35"))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)

    fig.tight_layout()
    base = OUTDIR / "02_motif_vs_tf_NR3C1_rank_pct"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)
    print(f"[drug_mech.02] wrote {base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
