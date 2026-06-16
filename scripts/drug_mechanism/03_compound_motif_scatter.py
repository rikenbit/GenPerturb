#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from scipy.stats import spearmanr

try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False


FOCAL_COLOR     = "black"
NON_FOCAL_COLOR = "#BFBFBF"
SEP_COLOR       = "#888888"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _common import (  # noqa: E402
    PAPER_RCPARAMS,
    load_drug_mech_data,
    MOTIF_TF_FAMILY,
)
from _rank_utils import rank_percentile_matrix  # noqa: E402

plt.rcParams.update(PAPER_RCPARAMS)

STUDY_FULL = "JialongJiang2024_CD8T_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
OUTDIR = ROOT / "figures" / STUDY_FULL / "drug_mechanism"
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "drug_mechanism"
DATA_DIR.mkdir(parents=True, exist_ok=True)

COMPOUNDS = [
    ("Jialong.Curcumin_CD3",         "Curcumin",        "NRF2 natural positive control"),
    ("Jialong.Rutaecarpine_CD3",     "Rutaecarpine",    "NRF2 natural (test/53)"),
    ("Jialong.Sophoricoside_CD3",    "Sophoricoside",    "candidate: NF-κB inhibitor"),
    ("Jialong.Isoalantolactone_CD3", "Isoalantolactone", "candidate: NF-κB inhibitor"),
]

TARGET_MOTIFS = {
    "NRF2": [
        "Nfe2l2", "MAFK", "Mafg", "MAFF", "MAF", "Mafb",
        "MAF::NFE2", "MAFG::NFE2L1", "Bach1::Mafk",
    ],
    "NF-kB": ["REL", "RELA"],
    "mTOR/FOXO": ["Foxo1", "Foxo3", "FOXO4"],
}

TARGET_COLORS = {
    "NRF2":      "#2E7D32",
    "NF-kB":     "#1F4E79",
    "mTOR/FOXO": "#7B1FA2",
}
OTHER_COLOR = "#9AA0A6"


def role_to_class(role: str) -> str | None:
    role_l = role.lower()
    if "nrf2" in role_l:
        return "NRF2"
    if "nf-κb" in role_l or "nf-kb" in role_l or "nfkb" in role_l:
        return "NF-kB"
    if "mtor" in role_l or "foxo" in role_l:
        return "mTOR/FOXO"
    return None


def family_signed_log2fc(fc_row: pd.Series, tf_symbols: list[str]) -> float:
    avail = [g for g in tf_symbols if g in fc_row.index]
    if not avail:
        return np.nan
    sub = fc_row.reindex(avail)
    if not np.isfinite(sub.values).any():
        return np.nan
    idx_max = sub.abs().idxmax()
    return float(sub.loc[idx_max])


def motif_to_genes(motif: str, fc_index_set: set) -> list[str | None]:
    parts = motif.split("::")
    out: list[str | None] = []
    for p in parts:
        if p in fc_index_set:
            out.append(p)
        elif p.upper() in fc_index_set:
            out.append(p.upper())
        else:
            out.append(None)
    return out


def build_split_entries(pert: str,
                        motif_universe: list[str],
                        rank_pct: pd.DataFrame,
                        fc_row: pd.Series,
                        fc_index_set: set):
    entries = []
    for m in motif_universe:
        y = float(rank_pct.loc[m, pert])
        if not np.isfinite(y):
            continue
        gene_parts = motif_to_genes(m, fc_index_set)
        if all(g is None for g in gene_parts):
            continue
        is_split = len(gene_parts) > 1
        for gi, g in enumerate(gene_parts):
            if g is None:
                continue
            x = float(fc_row[g])
            if not np.isfinite(x):
                continue
            entries.append({
                "motif": m,
                "gene": g,
                "focal_idx": gi,
                "x": x,
                "y": y,
                "is_split": is_split,
            })
    return entries


def _make_label_box(motif_name: str, focal_idx: int,
                     is_target: bool) -> HPacker:
    parts = motif_name.split("::")
    base_color = "black" if is_target else "#444"
    children = []
    for i, p in enumerate(parts):
        if len(parts) == 1:
            color = base_color
        else:
            color = FOCAL_COLOR if i == focal_idx else NON_FOCAL_COLOR
        children.append(
            TextArea(p.upper(),
                      textprops=dict(fontsize=10, color=color))
        )
        if i < len(parts) - 1:
            children.append(
                TextArea("::",
                          textprops=dict(fontsize=10, color=SEP_COLOR))
            )
    return HPacker(children=children, pad=0, sep=0, align="baseline")


def _place_labels_on_rail(y_axes_frac,
                            min_gap: float = 0.06,
                            y_min: float = 0.04,
                            y_max: float = 0.96):
    n = len(y_axes_frac)
    if n == 0:
        return []
    arr = np.asarray(y_axes_frac, dtype=float)
    order = np.argsort(-arr)            # descending
    sorted_y = np.clip(arr[order], y_min, y_max).astype(float)
    # Top-down pass: enforce min_gap.
    for i in range(1, n):
        if sorted_y[i - 1] - sorted_y[i] < min_gap:
            sorted_y[i] = sorted_y[i - 1] - min_gap
    if sorted_y[-1] < y_min:
        sorted_y[-1] = y_min
        for i in range(n - 2, -1, -1):
            if sorted_y[i] - sorted_y[i + 1] < min_gap:
                sorted_y[i] = sorted_y[i + 1] + min_gap
        if sorted_y[0] > y_max:
            sorted_y = np.linspace(y_max, y_min, n)
    rev = np.empty(n, dtype=int)
    rev[order] = np.arange(n)
    return sorted_y[rev].tolist()


def label_split_entries(ax,
                         motif_names: list[str],
                         focal_idx_arr: np.ndarray,
                         x: np.ndarray, y: np.ndarray,
                         is_split: np.ndarray,
                         is_target: np.ndarray,
                         n_top_other: int = 0,
                         rail_x_frac: float = 1.04,
                         rail_min_gap: float = 0.06):
    chosen = sorted(set(np.where(is_target)[0].tolist()))
    if n_top_other > 0:
        other_idx = np.where(~is_target)[0]
        if other_idx.size > 0:
            order = other_idx[np.argsort(-y[other_idx])]
            for ii in order[:n_top_other]:
                chosen.append(int(ii))
        chosen = sorted(set(chosen))

    if not chosen:
        return

    y0, y1 = ax.get_ylim()
    chosen_y_frac = [(y[i] - y0) / (y1 - y0) for i in chosen]
    label_ys = _place_labels_on_rail(chosen_y_frac, min_gap=rail_min_gap)

    for k, i in enumerate(chosen):
        label_box = _make_label_box(
            motif_names[i], int(focal_idx_arr[i]), bool(is_target[i])
        )
        ab = AnnotationBbox(
            label_box, (x[i], y[i]),
            xybox=(rail_x_frac, label_ys[k]),
            xycoords="data", boxcoords="axes fraction",
            box_alignment=(0.0, 0.5),
            frameon=False, pad=0.2, zorder=6,
            arrowprops=dict(arrowstyle="-", color="#888",
                             lw=0.5, shrinkB=3),
        )
        ax.add_artist(ab)


def main():
    print("[drug_mech.03] loading data ...")
    data = load_drug_mech_data()
    tfm_abs  = data["tfm_abs"]
    fc_df    = data["fc_pseudo_df"]
    SIG_THRESHOLD = -np.log10(0.05)
    tfm_sig = tfm_abs.where(tfm_abs >= SIG_THRESHOLD, 0.0)
    rank_pct = rank_percentile_matrix(
        tfm_sig, exclude_zero=True, method="min",
    )
    motif_universe = list(rank_pct.index)
    fc_index_set = set(fc_df.columns)
    print(f"[drug_mech.03]   motif universe size: {len(motif_universe)}; "
          f"per-compound q<0.05 cutoff: -log10(qval) >= {SIG_THRESHOLD:.3f}")

    summary_rows = []

    for pert, label, role in COMPOUNDS:
        if pert not in rank_pct.columns:
            print(f"[drug_mech.03]   SKIP {label}: pert not in rank_pct matrix")
            summary_rows.append({
                "perturbation": pert, "label": label, "role": role,
                "in_tfm_matrix": False,
                "n_motifs_plotted": 0,
                "n_target_motifs_present": 0,
                "spearman_rho": np.nan, "spearman_p": np.nan,
            })
            continue
        if pert not in fc_df.index:
            print(f"[drug_mech.03]   SKIP {label}: pert not in fc_df")
            continue

        target_class = role_to_class(role)
        target_set = set(TARGET_MOTIFS.get(target_class, []))
        color_target = TARGET_COLORS.get(target_class, "#C0392B")

        fc_row = fc_df.loc[pert]
        entries = build_split_entries(
            pert, motif_universe, rank_pct, fc_row, fc_index_set,
        )
        if not entries:
            print(f"[drug_mech.03]   SKIP {label}: no usable motif dots")
            continue

        motif_names   = [e["motif"] for e in entries]
        gene_names    = [e["gene"] for e in entries]
        focal_idx_arr = np.array([e["focal_idx"] for e in entries])
        x             = np.array([e["x"] for e in entries])
        y             = np.array([e["y"] for e in entries])
        is_split_arr  = np.array([e["is_split"] for e in entries])

        is_target = np.array([m in target_set for m in motif_names],
                              dtype=bool)
        n_target_dots = int(is_target.sum())
        unique_motifs = sorted(set(motif_names))
        unique_targets = sorted({m for m in unique_motifs if m in target_set})

        rho, p_rho = spearmanr(x, y)

        fig, ax = plt.subplots(figsize=(11.0, 7.0))
        ax.scatter(x[~is_target], y[~is_target], s=55, color=OTHER_COLOR,
                   alpha=0.55, edgecolor="gray", lw=0.2, zorder=2,
                   label=f"other TF dots (n={(~is_target).sum()})")
        if is_target.any():
            ax.scatter(x[is_target], y[is_target], s=140, color=color_target,
                       alpha=0.95, edgecolor="black", lw=0.7, zorder=4,
                       label=f"{target_class} family dots (n={n_target_dots})")

        ax.axvline(0, color="black", lw=0.4, ls=":")
        ax.set_xlabel(r"TF mRNA $\log_{2}$ fold change")
        ax.set_ylabel("motif rank percentile\n"
                       "(1 = top motif in compound)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{label}  —  {role}", fontsize=18, pad=8)
        ax.grid(linestyle=":", lw=0.4, alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=11)

        fig.subplots_adjust(left=0.085, right=0.70,
                              top=0.91, bottom=0.12)

        label_split_entries(
            ax, motif_names, focal_idx_arr,
            x, y, is_split_arr, is_target,
            n_top_other=0,
        )

        slug = label.replace(" ", "_")
        base = OUTDIR / f"03_motif_scatter_{slug}"
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(base.with_suffix(".png"))
        plt.close(fig)
        print(f"[drug_mech.03]   -> {base.with_suffix('.svg')}  "
              f"({target_class}, target motifs present: {len(unique_targets)},  "
              f"target dots: {n_target_dots})")

        seen: set[str] = set()
        order_by_y = np.argsort(-y)
        top_rows = []
        for idx in order_by_y:
            m = motif_names[idx]
            if m in seen:
                continue
            seen.add(m)
            same_motif = [k for k in range(len(entries))
                          if motif_names[k] == m]
            genes_for_m = sorted({gene_names[k] for k in same_motif})
            per_gene = ",".join(
                f"{gene_names[k]}={x[k]:+.3f}" for k in same_motif
            )
            top_rows.append({
                "rank":               len(top_rows) + 1,
                "motif":              m,
                "tf_genes_mapped":    ",".join(genes_for_m),
                "per_gene_log2fc":    per_gene,
                "motif_rank_pct":     float(y[idx]),
                "motif_neglog10_q":   float(tfm_abs.loc[m, pert]),
                "is_target":          bool(m in target_set),
            })
            if len(top_rows) >= 25:
                break
        pd.DataFrame(top_rows).to_csv(
            DATA_DIR / f"03_motif_scatter_{slug}_top_table.tsv",
            sep="\t", index=False, float_format="%.5g",
        )

        summary_rows.append({
            "perturbation":            pert,
            "label":                   label,
            "role":                    role,
            "target_class":            target_class,
            "in_tfm_matrix":           True,
            "n_motifs_unique":         len(unique_motifs),
            "n_dots_plotted":          int(x.size),
            "n_target_motifs_unique":  len(unique_targets),
            "n_target_dots":           n_target_dots,
            "spearman_rho":            float(rho),
            "spearman_p":              float(p_rho),
        })

    pd.DataFrame(summary_rows).to_csv(
        DATA_DIR / "03_motif_scatter_summary.tsv",
        sep="\t", index=False, float_format="%.5g",
    )

    print(f"[drug_mech.03] done. SVG/PNG in {OUTDIR}; TSV in {DATA_DIR}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
