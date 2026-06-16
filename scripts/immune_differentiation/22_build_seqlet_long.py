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

warnings.filterwarnings("ignore", category=FutureWarning)

CWD = Path(__file__).resolve().parents[2]
STUDY_FULL = (
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "__alphagenome_transfer_epoch100_batch256_adamw5e3"
)
SIG_SCORES_PATH = CWD / f"figures/{STUDY_FULL}/gene_signature/signature_scores.txt"
JASPAR_CLUSTERS = CWD / "reference/jaspar/clusters.tab"
MUTATION_DIR = CWD / f"figures/{STUDY_FULL}/seqlet_mutation/mutation_predictions"

OUT_DIR = CWD / f"figures/{STUDY_FULL}/seqlet_mutation"
RESULTS_DIR = OUT_DIR
FIG_DIR = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

LINEAGE_COLOR = {
    "Erythroid":  "#E8A09F", 
    "Granulocyte": "#9DBFD9", 
    "other":       "#CFCFCF", 
}

CLUSTER_LABEL = {
    "cluster_011_KLF": "KLF family",
    "cluster_011_SP":  "SP/GC-box family",
    "cluster_014":     "GATA family",
    "cluster_048":     "CEBP/ATF family",
    "cluster_041":     "STAT family",
}
CLUSTER_COLOR = {
    "cluster_014":     "#D62728", 
    "cluster_048":     "#2CA02C",
    "cluster_011_KLF": "#1F77B4", 
    "cluster_011_SP":  "#17BECF", 
    "cluster_041":     "#9467BD",
    "Other":           "#7F7F7F",
}


def split_klf_sp(cluster: str, motif_gene: str) -> str:
    if cluster != "cluster_011":
        return cluster
    if str(motif_gene).strip().upper().startswith("KLF"):
        return "cluster_011_KLF"
    return "cluster_011_SP"

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "figure.titlesize": 24,
    "svg.fonttype": "none",
})


def load_real_signature() -> pd.DataFrame:
    df = pd.read_csv(SIG_SCORES_PATH, sep="\t")
    df = df[df["value_type"] == "real"].copy()
    df = df.set_index("Perturbation")
    return df


def load_motif_to_cluster() -> dict[str, str]:
    clusters = pd.read_csv(JASPAR_CLUSTERS, sep="\t")
    name_to_cluster: dict[str, str] = {}
    for _, row in clusters.iterrows():
        for n in row["name"].split(","):
            key = n.strip().upper()
            if key and key not in name_to_cluster:
                name_to_cluster[key] = row["cluster"]

    def lookup(motif_gene: str) -> str:
        return name_to_cluster.get(str(motif_gene).strip().upper(), "Other")

    return lookup


def load_mutation_long(motif_to_cluster) -> pd.DataFrame:
    rows = []
    for tsv in sorted(MUTATION_DIR.glob("*_mutation_predictions.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        if df.empty:
            continue
        df["dFC"] = df["mt_fc"] - df["wt_fc"]
        agg = (
            df.groupby([
                "gene", "chromosome", "core_genomic_start", "core_genomic_end",
                "motif_gene", "motif_match", "target_perturbation", "pert_lineage",
            ], as_index=False)
            .agg(dFC_mean=("dFC", "mean"),
                 dFC_std=("dFC", "std"),
                 n_seeds=("seed", "nunique"))
        )
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["cluster"] = out["motif_gene"].map(motif_to_cluster)
    out["cluster"] = [
        split_klf_sp(c, mg) for c, mg in zip(out["cluster"], out["motif_gene"])
    ]
    out["cluster_label"] = out["cluster"].map(
        lambda c: CLUSTER_LABEL.get(c, c if c == "Other" else f"Other ({c})"),
    )
    return out


def lineage_for_pert(pert: str) -> str:
    if pert in ERYTHROID_PERTS:
        return "Erythroid"
    if pert in GRANULOCYTE_PERTS:
        return "Granulocyte"
    return "other"


def build_perturbation_table(real_sig: pd.DataFrame, signature: str) -> pd.DataFrame:
    nt = real_sig.loc["NT", signature] if "NT" in real_sig.index else 0.0
    df = real_sig[[signature]].copy()
    df = df.rename(columns={signature: "sig_real"})
    df["delta_sig"] = df["sig_real"] - nt
    df = df.drop(index="NT", errors="ignore")
    df["target_perturbation"] = "Norman." + df.index.astype(str)
    df["lineage"] = df["target_perturbation"].map(lineage_for_pert)
    return df


def plot_signature(
    signature: str,
    pert_table: pd.DataFrame,
    seqlet_long: pd.DataFrame,
    out_path: Path,
) -> dict:
    n_sig_genes = len(SIGNATURE_GENES[signature])
    target_genes = set(SIGNATURE_GENES[signature])

    sl = seqlet_long[seqlet_long["gene"].isin(target_genes)].copy()
    sl["dSig"] = sl["dFC_mean"] / float(n_sig_genes)

    pert_table = pert_table.sort_values("sig_real", ascending=False).copy()
    pert_order = list(pert_table["target_perturbation"])
    x_pos = np.arange(len(pert_order))
    pert_to_x = {p: i for i, p in enumerate(pert_order)}

    sl = sl[sl["target_perturbation"].isin(pert_to_x)].copy()
    sl["x_base"] = sl["target_perturbation"].map(pert_to_x)

    bar_means = (
        sl.groupby("target_perturbation")["dSig"].mean()
        .reindex(pert_order)
        .fillna(0.0)
    )
    pert_table = pert_table.assign(bar_value=bar_means.values)

    plot_inches = max(20.0, 0.18 * len(pert_order) + 6)
    legend_inches = 4.5 
    fig_h = 9.0
    fig_w = plot_inches + legend_inches
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_colors = [LINEAGE_COLOR[ln] for ln in pert_table["lineage"].values]
    edge_colors = [
        ("#7B2C2C" if ln == "Erythroid"
         else "#264F73" if ln == "Granulocyte"
         else "#888888")
        for ln in pert_table["lineage"].values
    ]
    bars = ax.bar(
        x_pos, pert_table["bar_value"].values,
        color=bar_colors, edgecolor=edge_colors, linewidth=0.6,
        width=0.78, alpha=0.85, zorder=1,
    )

    bar_lo = float(pert_table["bar_value"].min())
    bar_hi = float(pert_table["bar_value"].max())
    dot_lo = float(np.nanquantile(sl["dSig"], 0.02)) if len(sl) else bar_lo
    dot_hi = float(np.nanquantile(sl["dSig"], 0.98)) if len(sl) else bar_hi
    y_lo = min(bar_lo, dot_lo)
    y_hi = max(bar_hi, dot_hi)
    span = y_hi - y_lo if y_hi > y_lo else max(abs(y_hi), 1e-6)
    y_lo -= span * 0.10
    y_hi += span * 0.10

    cluster_counts = sl["cluster"].value_counts()
    cluster_order = list(cluster_counts.index)

    rng = np.random.default_rng(0)
    n_clusters = max(len(cluster_order), 1)
    spread = 0.22 
    cluster_means_per_x: dict[str, np.ndarray] = {}

    for ci, cluster in enumerate(cluster_order):
        sub = sl[sl["cluster"] == cluster]
        if sub.empty:
            continue

        if n_clusters > 1:
            sub_offset = (ci - (n_clusters - 1) / 2.0) * (0.55 / n_clusters)
        else:
            sub_offset = 0.0
        jitter = rng.uniform(-0.10, 0.10, size=len(sub))
        x_jit = sub["x_base"].to_numpy() + sub_offset + jitter

        y_vals = np.clip(sub["dSig"].to_numpy(), y_lo, y_hi)

        ax.scatter(
            x_jit, y_vals,
            s=42,
            color=CLUSTER_COLOR.get(cluster, "#444444"),
            alpha=0.85,
            edgecolors="white", linewidths=0.4,
            label=CLUSTER_LABEL.get(cluster, cluster),
            zorder=3,
        )

        per_x_mean = (
            sub.groupby("x_base")["dSig"].mean()
            .reindex(x_pos)
        )
        cluster_means_per_x[cluster] = per_x_mean.to_numpy()

    for cluster, means in cluster_means_per_x.items():
        valid = ~np.isnan(means)
        if valid.sum() < 2:
            continue
        ax.plot(
            x_pos[valid], means[valid],
            color=CLUSTER_COLOR.get(cluster, "#444444"),
            lw=1.6, alpha=0.85,
            marker="", zorder=2,
        )

    ax.axhline(0, color="black", lw=0.8, zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [p.replace("Norman.", "") for p in pert_order],
        rotation=90, ha="center",
    )
    ax.set_ylabel(f"Δ{signature} signature contribution (ΔFC / N, N={n_sig_genes})")
    ax.set_xlabel(f"Perturbation (sorted left→right by real {signature} score, descending)")
    ax.set_title(
        f"{signature} signature — seqlet-mutation cancellation per perturbation\n"
        f"bar = mean dot value per perturbation; "
        f"dot = (mt_fc - wt_fc) / {n_sig_genes} per seqlet (fold-change cancels ctrl-side mutation effect)",
    )
    ax.set_xlim(-0.7, len(pert_order) - 0.3)
    ax.set_ylim(y_lo, y_hi)

    bar_handles = [
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=LINEAGE_COLOR["Erythroid"],
               markeredgecolor="#7B2C2C", markersize=12,
               label=f"Erythroid pert (n={(pert_table['lineage']=='Erythroid').sum()})"),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=LINEAGE_COLOR["Granulocyte"],
               markeredgecolor="#264F73", markersize=12,
               label=f"Granulocyte pert (n={(pert_table['lineage']=='Granulocyte').sum()})"),
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=LINEAGE_COLOR["other"],
               markeredgecolor="#888888", markersize=12,
               label=f"Other (n={(pert_table['lineage']=='other').sum()})"),
    ]
    cluster_handles = [
        Line2D([0], [0], marker="o", linestyle="-",
               color=CLUSTER_COLOR.get(c, "#444444"),
               markerfacecolor=CLUSTER_COLOR.get(c, "#444444"),
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=9, lw=1.6,
               label=f"{CLUSTER_LABEL.get(c, c)} "
                     f"(N={(sl['cluster']==c).sum()} seqlet-pert)")
        for c in cluster_order
    ]
    leg1 = ax.legend(
        handles=bar_handles, title="Bar (perturbation lineage)",
        loc="upper left", bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0, framealpha=0.95,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=cluster_handles, title="Dot/line (motif JASPAR cluster)",
        loc="upper left", bbox_to_anchor=(1.005, 0.62),
        borderaxespad=0.0, framealpha=0.95,
    )

    ax.grid(axis="y", alpha=0.25)

    left_in, right_in, top_in, bottom_in = 1.2, legend_inches, 1.0, 2.6
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1.0 - right_in / fig_w,
        top=1.0 - top_in / fig_h,
        bottom=bottom_in / fig_h,
    )
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight",
                bbox_extra_artists=(leg1, leg2))
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200,
                bbox_extra_artists=(leg1, leg2))
    plt.close(fig)

    return {
        "n_perturbations": len(pert_order),
        "n_seqlet_pert_pairs": int(len(sl)),
        "clusters": {
            CLUSTER_LABEL.get(c, c): int((sl["cluster"] == c).sum())
            for c in cluster_order
        },
        "n_sig_genes": n_sig_genes,
    }


def main() -> None:
    print("Loading data…")
    real_sig = load_real_signature()
    print(f"  signature_scores.txt: {len(real_sig)} perturbations")

    motif_to_cluster = load_motif_to_cluster()
    seqlet_long = load_mutation_long(motif_to_cluster)
    print(f"  seqlet long table: {len(seqlet_long)} rows "
          f"(seqlet × perturbation, seed-aggregated)")

    seqlet_long_out = RESULTS_DIR / "seqlet_signature_delta_long.tsv"
    seqlet_long.to_csv(seqlet_long_out, sep="\t", index=False)

    summaries = {}
    for signature in ("Erythroid", "Granulocyte"):
        print(f"\n=== {signature} ===")
        pert_table = build_perturbation_table(real_sig, signature)
        pert_table.to_csv(
            RESULTS_DIR / f"perturbation_signature_delta_{signature.lower()}.tsv",
            sep="\t",
        )

        out_path = FIG_DIR / f"cancellation_{signature}"
        stats = plot_signature(signature, pert_table, seqlet_long, out_path)
        summaries[signature] = stats
        print(f"  perturbations plotted: {stats['n_perturbations']}")
        print(f"  seqlet × perturbation dots: {stats['n_seqlet_pert_pairs']}")
        for cl, n in stats["clusters"].items():
            print(f"    {cl}: {n}")

        focused = pert_table[pert_table["lineage"] != "other"].copy()
        if not focused.empty:
            f_path = FIG_DIR / f"cancellation_{signature}_lineage_only"
            f_stats = plot_signature(signature, focused, seqlet_long, f_path)
            print(f"  [focused] perturbations: {f_stats['n_perturbations']}")

    pd.DataFrame([
        {"signature": s, **{k: v for k, v in stats.items() if k != "clusters"},
         **{f"n_{cl}": n for cl, n in stats["clusters"].items()}}
        for s, stats in summaries.items()
    ]).to_csv(RESULTS_DIR / "run_summary.tsv", sep="\t", index=False)

    print("\nFigures →", FIG_DIR)


if __name__ == "__main__":
    main()
