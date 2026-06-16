#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _common import PAPER_RCPARAMS, load_drug_mech_data  # noqa: E402
from _glucocorticoid_pubchem import (  # noqa: E402
    PUBCHEM_GR_SCREEN_TSV,
    load_pubchem_gr_set,
)

plt.rcParams.update(PAPER_RCPARAMS)

STUDY_FULL = "JialongJiang2024_CD8T_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
OUTDIR = ROOT / "figures" / STUDY_FULL / "drug_mechanism"
OUTDIR.mkdir(parents=True, exist_ok=True)

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
    print("[drug_mech.01] loading data ...")
    gr_set     = load_pubchem_gr_set()
    print(f"[drug_mech.01]   GR set input = {PUBCHEM_GR_SCREEN_TSV}")
    data = load_drug_mech_data()
    tfm_z      = data["tfm_z"]
    drug_names = data["drug_names"]

    if MOTIF not in tfm_z.index:
        raise RuntimeError(f"{MOTIF} missing from tfm_z matrix")

    in_class = drug_names.isin(gr_set).values
    vals = tfm_z.loc[MOTIF].values.astype(float)
    print(f"[drug_mech.01]   n compounds = {len(vals)};  "
          f"in_class = {in_class.sum()};  out = {(~in_class).sum()}")

    p_wilcox = ranksums(vals[in_class], vals[~in_class],
                        alternative="greater").pvalue
    print(f"[drug_mech.01]   Wilcoxon one-sided {fmt_p(p_wilcox)}")

    rng = np.random.default_rng(42)
    x_oth = 0 + rng.uniform(-0.07, 0.07, size=int((~in_class).sum()))
    x_in  = 1 + rng.uniform(-0.07, 0.07, size=int(in_class.sum()))

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.scatter(x_oth, vals[~in_class], s=42, color=OTHER_COLOR,
               alpha=0.65, edgecolor="gray", lw=0.3, zorder=2)
    ax.scatter(x_in,  vals[in_class],  s=130, color=GR_COLOR,
               alpha=0.95, edgecolor="black", lw=0.7, zorder=3)
    ax.plot([-0.32, 0.32], [np.nanmedian(vals[~in_class])] * 2,
            color="black", lw=2.0, zorder=4)
    ax.plot([0.68, 1.32], [np.nanmedian(vals[in_class])] * 2,
            color=GR_COLOR, lw=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other compounds", "Corticosteroid (GR)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("cross-compound\nz-score")
    ax.set_title(MOTIF, pad=8, fontweight="bold")
    ax.grid(axis="y", linestyle=":", lw=0.4, alpha=0.4)

    ax.text(0.02, 0.98, fmt_p(p_wilcox),
            transform=ax.transAxes,
            fontsize=14, va="top", ha="left",
            family="DejaVu Sans Mono")

    fig.tight_layout()
    base = OUTDIR / "01_glucocorticoid_zscore_NR3C1"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)
    print(f"[drug_mech.01] wrote {base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
