#!/usr/bin/env python3
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

CWD = Path(__file__).resolve().parents[2]
STUDY = "NormanWeissman2019_filtered_mixscape_exnp_train"
SUFFIX = "alphagenome_transfer_epoch100_batch256_adamw5e3"
STUDY_FULL = f"{STUDY}__{SUFFIX}"

EXPR_PATH = CWD / f"data/{STUDY}.tsv"
MOTIF_MATRIX_POS_PATH = (
    CWD / f"attribution_analysis/tfmodisco/{STUDY_FULL}/tfmodisco_motif_matrix_pos.tsv"
)
MOTIF_MATRIX_NEG_PATH = (
    CWD / f"attribution_analysis/tfmodisco/{STUDY_FULL}/tfmodisco_motif_matrix_neg.tsv"
)
SIG_SCORES_PATH = CWD / f"figures/{STUDY_FULL}/gene_signature/signature_scores.txt"

FIG_BASE = CWD / f"figures/{STUDY_FULL}/signature_axis"
OUT_DIR = FIG_BASE
FIG_DIR = FIG_BASE
FIG_BASE.mkdir(parents=True, exist_ok=True)

ERYTHROID_PERTURBATIONS: set[str] = {
    "CBL_CNN1", "CBL_PTPN12", "CBL_PTPN9", "CBL_UBASH3B",
    "SAMD1_PTPN12", "SAMD1_UBASH3B",
    "UBASH3B_CNN1", "UBASH3B_PTPN12", "UBASH3B_PTPN9",
    "UBASH3B_UBASH3A", "UBASH3B_ZBTB25",
    "BPGM_SAMD1", "PTPN1",
    "PTPN12_PTPN9", "PTPN12_UBASH3A", "PTPN12_ZBTB25",
}
GRANULOCYTE_PERTURBATIONS: set[str] = {
    "SPI1", "CEBPA", "CEBPB",
    "CEBPE_CEBPA", "CEBPE_RUNX1T1", "CEBPE_SPI1",
    "CEBPE", "ETS2_CEBPE", "KLF1_CEBPA", "FOSB_CEBPE",
}

ERYTHROID_TFS: set[str] = {
    "GATA1", "KLF1", "TAL1", "LMO2", "NFE2",
    "FOG1", "ZFPM1", "BCL11A",
    "GATA1::TAL1", "TAL1::TCF3",
}
GRANULOCYTE_TFS: set[str] = {
    "CEBPA", "SPI1", "Spi1", "CEBPB", "CEBPE",
    "GFI1", "IRF4", "IRF8", "BATF3", "KLF4", "MAFB",
}

LINEAGE_COLOR = {
    "Erythroid": "#C0392B",  
    "Granulocyte": "#2C7FB8",
    "other": "#9E9E9E",  
}

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
    "svg.fonttype": "none",
})

ANNOT_FONTSIZE = 16 


def load_signature_scores() -> pd.DataFrame:
    df = pd.read_csv(SIG_SCORES_PATH, sep="\t")
    return df


def load_expression() -> pd.DataFrame:
    df = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    df = df.drop(columns=["training"], errors="ignore")
    df.columns = [c.replace("Norman.", "") for c in df.columns]
    return df


def load_motif_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [c.replace("Norman.", "") for c in df.columns]
    return df


def load_motif_matrix_pos() -> pd.DataFrame:
    return load_motif_matrix(MOTIF_MATRIX_POS_PATH)


def load_motif_matrix_neg() -> pd.DataFrame:
    return load_motif_matrix(MOTIF_MATRIX_NEG_PATH)


def get_signature_table(sig_df: pd.DataFrame, signature: str) -> pd.DataFrame:
    real = (
        sig_df[sig_df["value_type"] == "real"]
        .set_index("Perturbation")[signature]
        .rename("observed")
    )
    pred = (
        sig_df[sig_df["value_type"] == "pred"]
        .set_index("Perturbation")[signature]
        .rename("predicted")
    )
    out = pd.concat([real, pred], axis=1).dropna()
    out = out.drop(index="NT", errors="ignore")
    return out


def lineage_for_perturbation(pert: str) -> str:
    if pert in ERYTHROID_PERTURBATIONS:
        return "Erythroid"
    if pert in GRANULOCYTE_PERTURBATIONS:
        return "Granulocyte"
    return "other"


def _lineage_of_token(token: str) -> str:
    """Classify a single TF gene token (case-insensitive)."""
    ery_upper = {t.upper() for t in ERYTHROID_TFS}
    gran_upper = {t.upper() for t in GRANULOCYTE_TFS}
    t = token.strip().upper()
    if t in ery_upper:
        return "Erythroid"
    if t in gran_upper:
        return "Granulocyte"
    return "other"


def lineage_for_motif(motif: str) -> str:
    parts = [p.strip() for p in motif.split("::")]
    lineages = {_lineage_of_token(p) for p in parts}
    if "Erythroid" in lineages:
        return "Erythroid"
    if "Granulocyte" in lineages:
        return "Granulocyte"
    return "other"


def lineage_for_motif_tf(motif: str, tf_gene: str) -> str:
    return _lineage_of_token(tf_gene)


def motif_to_tf_genes(motif: str, expressed_genes: set[str]) -> list[str]:
    upper_to_orig = {g.upper(): g for g in expressed_genes}
    out: list[str] = []
    seen: set[str] = set()
    for part in motif.split("::"):
        key = part.strip().upper()
        gene = upper_to_orig.get(key)
        if gene is not None and gene not in seen:
            out.append(gene)
            seen.add(gene)
    return out


def _corr_motif_signature(
    motif_matrix: pd.DataFrame, sig_vec: np.ndarray, common: list[str],
    min_nonzero: int = 5,
) -> dict[str, tuple[float, float, int]]:
    out: dict[str, tuple[float, float, int]] = {}
    for motif in motif_matrix.index:
        vec = motif_matrix.loc[motif, common].astype(float).values
        nonzero = int((vec != 0).sum())
        if nonzero < min_nonzero or np.std(vec) == 0:
            continue
        r, p = pearsonr(vec, sig_vec)
        out[motif] = (r, p, nonzero)
    return out


def compute_signature_correlations(
    sig_table: pd.DataFrame,
    motif_pos: pd.DataFrame,
    motif_neg: pd.DataFrame,
    expr_matrix: pd.DataFrame,
    signature_value: str = "observed",
) -> pd.DataFrame:
    common = list(
        sig_table.index
        .intersection(motif_pos.columns)
        .intersection(motif_neg.columns)
        .intersection(expr_matrix.columns)
    )
    common = [p for p in common if p != "NT"]
    sig_vec = sig_table.loc[common, signature_value].astype(float).values

    pos_corr = _corr_motif_signature(motif_pos, sig_vec, common)
    neg_corr = _corr_motif_signature(motif_neg, sig_vec, common)

    expressed_genes = set(expr_matrix.index)
    all_motifs = set(pos_corr) | set(neg_corr)

    rows: list[dict] = []
    for motif in all_motifs:
        pos_entry = pos_corr.get(motif)
        neg_entry = neg_corr.get(motif)

        candidates = []
        if pos_entry is not None:
            candidates.append(("pos", *pos_entry))
        if neg_entry is not None:
            candidates.append(("neg", *neg_entry))
        if not candidates:
            continue
        # Pick the direction with the larger |r|
        direction, r_motif, p_motif, nonzero = max(
            candidates, key=lambda x: abs(x[1]),
        )

        tf_genes = motif_to_tf_genes(motif, expressed_genes)
        if not tf_genes:
            continue
        for tf_gene in tf_genes:
            expr_vec = expr_matrix.loc[tf_gene, common].astype(float).values
            if np.std(expr_vec) == 0:
                continue
            r_expr, p_expr = pearsonr(expr_vec, sig_vec)
            rows.append({
                "motif": motif,
                "tf_gene": tf_gene,
                "is_composite": "::" in motif,
                "n_components": len(tf_genes),
                "direction": direction,
                "r_motif_sig": r_motif,
                "p_motif_sig": p_motif,
                "r_expr_sig": r_expr,
                "p_expr_sig": p_expr,
                "n_nonzero_motif": nonzero,
                "n_perts": len(common),
                "lineage": lineage_for_motif_tf(motif, tf_gene),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    motif_table = df.drop_duplicates("motif")[["motif", "p_motif_sig"]].reset_index(drop=True)
    _, fdr_motif, _, _ = multipletests(motif_table["p_motif_sig"].fillna(1.0),
                                       method="fdr_bh")
    motif_table["fdr_motif_sig"] = fdr_motif
    df = df.merge(motif_table[["motif", "fdr_motif_sig"]], on="motif", how="left")
    _, fdr_expr, _, _ = multipletests(df["p_expr_sig"].fillna(1.0), method="fdr_bh")
    df["fdr_expr_sig"] = fdr_expr
    return df


def plot_B_observed_vs_predicted(
    sig_table: pd.DataFrame, signature: str, out_path: Path,
) -> dict:
    df = sig_table.copy()
    df["lineage"] = df.index.map(lineage_for_perturbation)

    fig, ax = plt.subplots(figsize=(11, 11))
    for lineage in ("other", "Granulocyte", "Erythroid"):  # plot lineage groups on top
        sub = df[df["lineage"] == lineage]
        ax.scatter(
            sub["observed"], sub["predicted"],
            s=80 if lineage == "other" else 150,
            color=LINEAGE_COLOR[lineage],
            alpha=0.6 if lineage == "other" else 0.95,
            edgecolors="black" if lineage != "other" else "none",
            linewidths=0.6,
            label=f"{lineage} (n={len(sub)})",
            zorder=2 if lineage == "other" else 3,
        )

    lo = float(min(df["observed"].min(), df["predicted"].min()))
    hi = float(max(df["observed"].max(), df["predicted"].max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="gray", lw=1, ls="--", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    pearson_r, pval = pearsonr(df["observed"].values, df["predicted"].values)
    ax.set_title(
        f"{signature} signature: observed vs predicted\n"
        f"Pearson r = {pearson_r:.3f} (p = {pval:.2e}); N = {len(df)}",
    )
    ax.set_xlabel(f"Observed {signature} signature score")
    ax.set_ylabel(f"Predicted {signature} signature score")

    label_df = df[df["lineage"] != "other"]
    texts = []
    for pert, row in label_df.iterrows():
        texts.append(ax.text(
            row["observed"], row["predicted"], pert,
            fontsize=ANNOT_FONTSIZE,
            color=LINEAGE_COLOR[row["lineage"]],
            fontweight="bold",
            ha="center", va="center", zorder=5,
        ))
    if texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.5, 1.8),
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7, alpha=0.85),
            force_text=(0.5, 0.7),
            force_static=(0.3, 0.5),
            min_arrow_len=3.0,
            ensure_inside_axes=True,
        )

    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    return {"pearson_r": pearson_r, "pearson_p": pval, "n": len(df)}


def plot_C_motif_vs_expr_correlation(
    corr_df: pd.DataFrame, signature: str, out_path: Path,
    sig_threshold: float = 0.05, threshold_kind: str = "fdr",
) -> dict:
    if threshold_kind == "fdr":
        sig_motifs = corr_df[corr_df["fdr_motif_sig"] < sig_threshold].copy()
        threshold_label = f"motif FDR < {sig_threshold:g}"
    else:
        sig_motifs = corr_df[corr_df["p_motif_sig"] < sig_threshold].copy()
        threshold_label = f"motif p < {sig_threshold:g}"

    if sig_motifs.empty:
        # Fall back to nominal p if FDR is too strict
        sig_motifs = corr_df[corr_df["p_motif_sig"] < 0.05].copy()
        threshold_label = "motif p < 0.05 (FDR yielded 0)"

    lineage_rows = sig_motifs[sig_motifs["lineage"].isin(("Erythroid", "Granulocyte"))].copy()
    lineage_rows["__label"] = np.where(
        lineage_rows["is_composite"],
        lineage_rows["tf_gene"].astype(str).str.upper(),
        lineage_rows["motif"].astype(str).str.upper(),
    )
    lineage_rows = lineage_rows.sort_values(
        ["lineage", "__label", "is_composite", "p_motif_sig"],
        ascending=[True, True, True, True],
    )
    lineage_rows = lineage_rows.drop_duplicates(subset=["lineage", "__label"], keep="first")
    other_rows = sig_motifs[sig_motifs["lineage"] == "other"]
    sig_motifs = pd.concat([other_rows, lineage_rows.drop(columns="__label")], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 11))
    for lineage in ("other", "Granulocyte", "Erythroid"):
        sub = sig_motifs[sig_motifs["lineage"] == lineage]
        ax.scatter(
            sub["r_motif_sig"], sub["r_expr_sig"],
            s=80 if lineage == "other" else 170,
            color=LINEAGE_COLOR[lineage],
            alpha=0.55 if lineage == "other" else 0.95,
            edgecolors="black" if lineage != "other" else "none",
            linewidths=0.6,
            label=f"{lineage} TFs (n={len(sub)})",
            zorder=2 if lineage == "other" else 3,
        )

    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.plot([-1, 1], [-1, 1], color="gray", lw=0.6, ls=":", alpha=0.5)

    overall_r, overall_p = pearsonr(
        sig_motifs["r_motif_sig"].values, sig_motifs["r_expr_sig"].values,
    )
    ax.set_title(
        f"{signature}: motif↔signature vs TF-expression↔signature\n"
        f"Pearson r = {overall_r:.3f} (p = {overall_p:.2e}); "
        f"N = {len(sig_motifs)} ({threshold_label})",
    )
    ax.set_xlabel(f"Pearson r (motif score, {signature} signature)")
    ax.set_ylabel(f"Pearson r (TF expression, {signature} signature)")

    lim = max(1.0, sig_motifs[["r_motif_sig", "r_expr_sig"]].abs().max().max() + 0.20)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)

    label_sub = sig_motifs[sig_motifs["lineage"].isin(("Erythroid", "Granulocyte"))]
    label_sub = label_sub.sort_values(
        ["is_composite", "p_motif_sig"], ascending=[True, True],
    )
    seen_labels: set[str] = set()
    texts = []
    for _, row in label_sub.iterrows():
        if row.get("is_composite", False):
            label = str(row["tf_gene"]).upper()
        else:
            label = str(row["motif"]).upper()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        texts.append(ax.text(
            row["r_motif_sig"], row["r_expr_sig"], label,
            fontsize=ANNOT_FONTSIZE,
            color=LINEAGE_COLOR[row["lineage"]],
            fontweight="bold",
            ha="center", va="center", zorder=5,
        ))
    if texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.5, 1.8),
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7, alpha=0.85),
            force_text=(0.5, 0.7),
            force_static=(0.3, 0.5),
            min_arrow_len=3.0,
            max_move=60,
            ensure_inside_axes=True,
        )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return {"n": len(sig_motifs), "pearson_r": overall_r, "pearson_p": overall_p,
            "threshold_label": threshold_label}


@dataclass
class SignatureRunSummary:
    signature: str
    B: dict
    C: dict


def run_for_signature(
    signature: str,
    sig_df: pd.DataFrame,
    motif_pos: pd.DataFrame,
    motif_neg: pd.DataFrame,
    expr_matrix: pd.DataFrame,
) -> SignatureRunSummary:
    print(f"\n=== {signature} ===")
    sig_table = get_signature_table(sig_df, signature)
    print(f"  N perturbations (real ∩ pred, NT excluded): {len(sig_table)}")

    suffix = signature.lower()

    B_path = FIG_DIR / f"B_obs_vs_pred_{suffix}"
    B_stats = plot_B_observed_vs_predicted(sig_table, signature, B_path)
    print(f"  B: r={B_stats['pearson_r']:.3f}, "
          f"p={B_stats['pearson_p']:.2e}, N={B_stats['n']}")

    corr_df = compute_signature_correlations(
        sig_table, motif_pos, motif_neg, expr_matrix, signature_value="observed",
    )
    corr_df.to_csv(OUT_DIR / f"C_correlations_{suffix}.tsv", sep="\t", index=False)
    print(f"  C: scored {len(corr_df)} motifs with mapped TF gene")

    C_path = FIG_DIR / f"C_motif_vs_expr_corr_{suffix}"
    C_stats = plot_C_motif_vs_expr_correlation(corr_df, signature, C_path,
                                                sig_threshold=0.05,
                                                threshold_kind="fdr")
    print(f"  C: plotted {C_stats['n']} significant motifs ({C_stats['threshold_label']})")

    return SignatureRunSummary(signature=signature, B=B_stats, C=C_stats)


def main() -> None:
    print("Loading data…")
    sig_df = load_signature_scores()
    expr_matrix = load_expression()
    motif_pos = load_motif_matrix_pos()
    motif_neg = load_motif_matrix_neg()
    print(f"  expr_matrix shape={expr_matrix.shape}")
    print(f"  motif_pos shape={motif_pos.shape}; motif_neg shape={motif_neg.shape}")

    summaries: list[SignatureRunSummary] = []
    for signature in ("Erythroid", "Granulocyte"):
        summaries.append(
            run_for_signature(signature, sig_df, motif_pos, motif_neg,
                              expr_matrix),
        )

    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "signature": s.signature,
            "B_n": s.B["n"],
            "B_pearson_r": s.B["pearson_r"],
            "B_pearson_p": s.B["pearson_p"],
            "C_n_significant": s.C["n"],
            "C_overall_r": s.C["pearson_r"],
            "C_overall_p": s.C["pearson_p"],
            "C_threshold": s.C["threshold_label"],
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "run_summary.tsv", sep="\t", index=False)
    print("\nDone. Figures and tables saved under", FIG_DIR)


if __name__ == "__main__":
    main()
