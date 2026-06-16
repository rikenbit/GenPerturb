#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

CWD = Path(__file__).resolve().parents[2]
STUDY_FULL = (
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "__alphagenome_transfer_epoch100_batch256_adamw5e3"
)
OUT_DIR = CWD / f"figures/{STUDY_FULL}/seqlet_mutation"
RESULTS_DIR = OUT_DIR
FIG_DIR = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQLET_LONG = RESULTS_DIR / "seqlet_signature_delta_long.tsv"
PERT_TABLES = {
    "Erythroid":   RESULTS_DIR / "perturbation_signature_delta_erythroid.tsv",
    "Granulocyte": RESULTS_DIR / "perturbation_signature_delta_granulocyte.tsv",
}

SIGNATURE_GENES = {
    "Erythroid":   ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "GYPA", "ERMAP"],
    "Granulocyte": ["ITGAM", "CSF3R", "LST1", "CD33"],
}

ERYTHROID_PERTS = {
    "Norman.CBL_CNN1", "Norman.CBL_PTPN12", "Norman.CBL_PTPN9",
    "Norman.CBL_UBASH3B", "Norman.SAMD1_PTPN12", "Norman.SAMD1_UBASH3B",
    "Norman.UBASH3B_CNN1", "Norman.UBASH3B_PTPN12", "Norman.UBASH3B_PTPN9",
    "Norman.UBASH3B_UBASH3A", "Norman.UBASH3B_ZBTB25", "Norman.BPGM_SAMD1",
    "Norman.PTPN1", "Norman.PTPN12_PTPN9", "Norman.PTPN12_UBASH3A",
    "Norman.PTPN12_ZBTB25",
}
GRANULOCYTE_PERTS = {
    "Norman.SPI1", "Norman.CEBPA", "Norman.CEBPB",
    "Norman.CEBPE_CEBPA", "Norman.CEBPE_RUNX1T1", "Norman.CEBPE_SPI1",
    "Norman.CEBPE", "Norman.ETS2_CEBPE", "Norman.KLF1_CEBPA",
    "Norman.FOSB_CEBPE",
}

TICK_COLOR = {
    "Erythroid":  "#C0392B", 
    "Granulocyte": "#2C7FB8",
    "other":       "#7F7F7F",
}

FAMILY_ORDER = [
    "GATA family",
    "CEBP/ATF family",
    "KLF family",
    "SP/GC-box family",
    "STAT family",
]
FAMILY_COLOR = {
    "GATA family":      "#D62728",
    "CEBP/ATF family":  "#2CA02C",
    "KLF family":       "#1F77B4",
    "SP/GC-box family": "#17BECF",
    "STAT family":      "#9467BD",
}

NEG_CONTROL_FAMILY = {
    "Erythroid":   {"CEBP/ATF family"},   # r ≈ -0.04, p ≈ 0.53
    "Granulocyte": {"STAT family"},       # r ≈ -0.04, p ≈ 0.59
}

Y_LIMITS = {
    "Erythroid":   (-0.015, 0.015),
    "Granulocyte": (-0.040, 0.020),
}

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
    "svg.fonttype": "none",
})


def lineage_for_pert(pert: str) -> str:
    if pert in ERYTHROID_PERTS:
        return "Erythroid"
    if pert in GRANULOCYTE_PERTS:
        return "Granulocyte"
    return "other"


def load_seqlet_long() -> pd.DataFrame:
    df = pd.read_csv(SEQLET_LONG, sep="\t")
    return df


def load_pert_table(signature: str) -> pd.DataFrame:
    df = pd.read_csv(PERT_TABLES[signature], sep="\t", index_col=0)
    df["lineage"] = df["target_perturbation"].map(lineage_for_pert)
    return df


def per_tf_correlation(seqlet_long: pd.DataFrame,
                       pert_table: pd.DataFrame,
                       signature: str) -> pd.DataFrame:
    target_genes = set(SIGNATURE_GENES[signature])
    N = len(SIGNATURE_GENES[signature])
    sub = seqlet_long[seqlet_long["gene"].isin(target_genes)].copy()
    sub["dSig"] = sub["dFC_mean"] / float(N)
    pivot = (
        sub.groupby(["target_perturbation", "motif_gene", "cluster_label"])
            ["dSig"].mean().reset_index()
    )
    pivot = pivot.merge(
        pert_table[["target_perturbation", "sig_real", "delta_sig"]],
        on="target_perturbation",
    )
    rows = []
    for (tf, fam), g in pivot.groupby(["motif_gene", "cluster_label"]):
        if len(g) < 5 or g["dSig"].std(ddof=0) == 0:
            continue
        r, p = pearsonr(g["dSig"].values, g["sig_real"].values)
        rho, prho = spearmanr(g["dSig"].values, g["sig_real"].values)
        rows.append({
            "motif_gene": tf, "family": fam, "n_perts": len(g),
            "pearson_r": r, "pearson_p": p,
            "spearman": rho, "spearman_p": prho,
            "mean_dSig": g["dSig"].mean(),
            "std_dSig": g["dSig"].std(),
        })
    df = pd.DataFrame(rows).sort_values("pearson_r")
    df["abs_r"] = df["pearson_r"].abs()
    return df


def per_family_correlation(seqlet_long: pd.DataFrame,
                           pert_table: pd.DataFrame,
                           signature: str) -> pd.DataFrame:
    target_genes = set(SIGNATURE_GENES[signature])
    N = len(SIGNATURE_GENES[signature])
    sub = seqlet_long[seqlet_long["gene"].isin(target_genes)].copy()
    sub["dSig"] = sub["dFC_mean"] / float(N)
    pivot = (
        sub.groupby(["target_perturbation", "cluster_label"])["dSig"]
            .mean().reset_index()
    )
    pivot = pivot.merge(
        pert_table[["target_perturbation", "sig_real", "delta_sig"]],
        on="target_perturbation",
    )
    rows = []
    for fam, g in pivot.groupby("cluster_label"):
        if len(g) < 5 or g["dSig"].std(ddof=0) == 0:
            continue
        r, p = pearsonr(g["dSig"].values, g["sig_real"].values)
        rho, prho = spearmanr(g["dSig"].values, g["sig_real"].values)
        rows.append({
            "family": fam, "n_perts": len(g),
            "pearson_r": r, "pearson_p": p,
            "spearman": rho, "spearman_p": prho,
            "mean_dSig": g["dSig"].mean(),
        })
    df = pd.DataFrame(rows)
    df["neg_control"] = df["pearson_p"].apply(
        lambda x: "YES (uncorrelated)" if x > 0.05 else "no",
    )
    return df.sort_values("pearson_r")


def plot_per_family(signature: str,
                    pert_table: pd.DataFrame,
                    seqlet_long: pd.DataFrame,
                    family_corr: pd.DataFrame,
                    out_path: Path) -> dict:
    target_genes = set(SIGNATURE_GENES[signature])
    N = len(SIGNATURE_GENES[signature])

    pt = pert_table[pert_table["lineage"] != "other"].copy()
    pt = pt.sort_values("sig_real", ascending=False).reset_index(drop=True)
    pert_order = list(pt["target_perturbation"])
    pert_to_x = {p: i for i, p in enumerate(pert_order)}
    x_pos = np.arange(len(pert_order))

    sl = seqlet_long[
        seqlet_long["gene"].isin(target_genes) &
        seqlet_long["target_perturbation"].isin(pert_to_x)
    ].copy()
    sl["dSig"] = sl["dFC_mean"] / float(N)
    sl["x_base"] = sl["target_perturbation"].map(pert_to_x)

    families = [f for f in FAMILY_ORDER if f in sl["cluster_label"].unique()]
    n_fam = len(families)
    bar_width = 0.72 / max(n_fam, 1)  # share the column

    fig_w = max(16.0, 0.55 * len(pert_order) + 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, 9.5))

    family_means_per_x: dict[str, np.ndarray] = {}
    for i, fam in enumerate(families):
        sub = sl[sl["cluster_label"] == fam]
        means = (
            sub.groupby("x_base")["dSig"].mean()
              .reindex(x_pos).fillna(0.0)
        )
        offset = (i - (n_fam - 1) / 2.0) * bar_width
        ax.bar(
            x_pos + offset, means.values,
            width=bar_width * 0.95,
            color=FAMILY_COLOR.get(fam, "#444444"),
            alpha=0.55, edgecolor=FAMILY_COLOR.get(fam, "#444444"),
            linewidth=0.6, zorder=1, label="_nolegend_",
        )
        family_means_per_x[fam] = means.values

    rng = np.random.default_rng(0)
    for i, fam in enumerate(families):
        sub = sl[sl["cluster_label"] == fam]
        if sub.empty:
            continue
        offset = (i - (n_fam - 1) / 2.0) * bar_width
        jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35, size=len(sub))
        ax.scatter(
            sub["x_base"].to_numpy() + offset + jitter,
            sub["dSig"].to_numpy(),
            s=24, color=FAMILY_COLOR.get(fam, "#444444"),
            alpha=0.85, edgecolors="white", linewidths=0.3,
            label="_nolegend_", zorder=3,
        )

    for fam, means in family_means_per_x.items():
        ax.plot(
            x_pos, means,
            color=FAMILY_COLOR.get(fam, "#444444"),
            lw=2.0, alpha=0.95, zorder=2,
        )

    ax.axhline(0, color="black", lw=0.8, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [p.replace("Norman.", "") for p in pert_order],
        rotation=70, ha="right",
    )
    for tick, p in zip(ax.get_xticklabels(), pert_order):
        tick.set_color(TICK_COLOR[lineage_for_pert(p)])
        tick.set_fontweight("bold")

    ax.set_xlim(-0.7, len(pert_order) - 0.3)
    ax.set_ylim(*Y_LIMITS[signature])  # clip outliers so bars are readable
    ax.set_ylabel(f"Δ{signature} signature contribution")
    ax.set_xlabel(f"Perturbation (sorted by real {signature} score, descending)")
    ax.set_title(f"{signature} signature — per-family seqlet-mutation cancellation")

    fam_handles = []
    for fam in families:
        ng = "  (neg-ctrl)" if fam in NEG_CONTROL_FAMILY[signature] else ""
        fam_handles.append(
            Line2D([0], [0], marker="s", linestyle="-",
                   color=FAMILY_COLOR.get(fam, "#444444"),
                   markerfacecolor=FAMILY_COLOR.get(fam, "#444444"),
                   markeredgecolor="white", markersize=12, lw=2.2,
                   label=f"{fam}{ng}"),
        )
    ax.legend(handles=fam_handles, title="Family",
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    return {
        "n_perts": len(pert_order),
        "families": families,
        "n_seqlet_pert_pairs": int(len(sl)),
    }


def main() -> None:
    seqlet_long = load_seqlet_long()
    print(f"Loaded seqlet long: {len(seqlet_long)} rows")

    summary_rows = []
    for signature in ("Erythroid", "Granulocyte"):
        print(f"\n=== {signature} ===")
        pert_table = load_pert_table(signature)

        tf_corr = per_tf_correlation(seqlet_long, pert_table, signature)
        tf_corr.to_csv(
            RESULTS_DIR / f"tf_correlation_{signature.lower()}.tsv",
            sep="\t", index=False,
        )
        print(f"  per-TF correlations → tf_correlation_{signature.lower()}.tsv "
              f"({len(tf_corr)} TFs)")

        fam_corr = per_family_correlation(seqlet_long, pert_table, signature)
        fam_corr.to_csv(
            RESULTS_DIR / f"cluster_correlation_{signature.lower()}.tsv",
            sep="\t", index=False,
        )
        print(f"  per-family correlations:")
        print(fam_corr.to_string(index=False))

        out_path = FIG_DIR / f"cancellation_per_family_{signature}"
        stats = plot_per_family(signature, pert_table, seqlet_long, fam_corr, out_path)
        print(f"  figure: {out_path}.svg/.png  "
              f"(perts={stats['n_perts']}, dots={stats['n_seqlet_pert_pairs']})")
        summary_rows.append({
            "signature": signature,
            "n_perts_lineage_only": stats["n_perts"],
            "n_seqlet_pert_pairs": stats["n_seqlet_pert_pairs"],
            "families_present": ",".join(stats["families"]),
            "neg_control_family": ",".join(NEG_CONTROL_FAMILY[signature]),
        })

    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "run_summary.tsv", sep="\t", index=False,
    )
    print("\nDone →", FIG_DIR)


if __name__ == "__main__":
    main()
