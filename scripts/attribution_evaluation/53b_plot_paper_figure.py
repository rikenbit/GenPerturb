#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

import h5py
import logomaker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CWD = Path(__file__).resolve().parents[2]
SCRIPT_DIR = CWD / "scripts" / "attribution_evaluation"

sys.path.insert(0, str(SCRIPT_DIR))
atac_utils = importlib.import_module("53_atac_utils")  # noqa: E402

STUDY_DIRNAME = "Martin_full"
STUDY_FULL = "MartinRufino2025_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3"

DATA_DIR = CWD / "attribution_analysis" / "insilico_mutation" / STUDY_DIRNAME / "paper_figure"
CANDIDATES_TSV = DATA_DIR / "candidates.tsv"

FIG_DIR = CWD / "figures" / STUDY_FULL / "paper_figure"

TSS_BED = CWD / "fasta" / "gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed"
GTF_PATH = CWD / "fasta" / "gencode.v46.chr_patch_hapl_scaff.basic.annotation.gtf"
BED_PATH = CWD / "fasta" / "MartinRufino2025_mixscape_exnp_train.bed"

ATTR_DIR = CWD / "attribution" / STUDY_FULL

ATAC_TSV = CWD / "data" / "MartinRufino2025_atac_cpm.tsv"
ATAC_CTRL_COL = "MartinRufino.NT"

MODISCO_DIR = CWD / "tfmodisco" / STUDY_FULL

CONTEXT_LENGTH = 1_048_576
HALF_CONTEXT = CONTEXT_LENGTH // 2
BIN_SIZE = 128
WINDOW_LEN = 96

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#009E73", "C": "#0072B2", "G": "#E69F00", "T": "#D55E00"}

PERT_COLOR = "#cc3333"
CTRL_COLOR = "#1f77b4"
DIFF_POS_COLOR = "#D55E00"
DIFF_NEG_COLOR = "#0072B2"
ATAC_POS_COLOR = "#444444"

BAR_COLORS = {
    "WT ctrl":  "#7fbcd9",
    "WT pert":  "#1f77b4",
    "Mut ctrl": "#f4a8a4",
    "Mut pert": "#cc3333",
}

BASE_FONT = 28
plt.rcParams.update({"font.size": BASE_FONT})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flank", type=int, default=None,
                   help="Force a fixed half-window (bp). If unset (default) "
                        "the window is sized dynamically per-candidate.")
    p.add_argument("--min-flank", type=int, default=4000,
                   help="Minimum half-window for the dynamic mode (default 4000).")
    p.add_argument("--margin", type=int, default=2000,
                   help="Extra bp on each side beyond the farther of "
                        "{TSS, seqlet} in dynamic mode (default 2000).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N candidates (debug)")
    p.add_argument("--gene-filter", default=None,
                   help="Only render this gene (debug)")
    return p.parse_args()


def _compute_view_window(tss_pos: int, core_mid: int,
                          fixed_flank: int | None,
                          min_flank: int, margin: int) -> tuple[int, int]:
    if fixed_flank is not None:
        return tss_pos - fixed_flank, tss_pos + fixed_flank
    centre = (tss_pos + core_mid) // 2
    half = max(min_flank, abs(core_mid - tss_pos) // 2 + margin)
    half = ((half + 999) // 1000) * 1000
    return centre - half, centre + half


def load_tss_bed():
    df = pd.read_csv(TSS_BED, sep="\t", header=None,
                     names=["chrom", "start", "end", "gene", "score", "strand", "fold"])
    out = {}
    for _, r in df.iterrows():
        out.setdefault(r["gene"], (r["chrom"], int(r["start"]), r["strand"]))
    return out


def load_training_bed_tss():
    df = pd.read_csv(BED_PATH, sep="\t", header=None,
                     names=["chrom", "start", "end", "gene", "score", "strand", "training"])
    return {r["gene"]: (r["chrom"], int(r["start"]), r["strand"]) for _, r in df.iterrows()}


def grep_gtf_for_genes(genes: list[str]) -> str:
    if not genes:
        return ""
    pattern = "|".join(f'gene_name "{g}"' for g in sorted(set(genes)))
    proc = subprocess.run(
        ["grep", "-E", pattern, str(GTF_PATH)],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


_GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')
_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')
_TX_NAME_RE = re.compile(r'transcript_name "([^"]+)"')


def parse_gene_models(gtf_text: str) -> dict[str, dict]:
    transcripts: dict[str, dict] = {}
    for line in gtf_text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 9:
            continue
        feat = cols[2]
        if feat not in ("transcript", "exon", "UTR", "CDS"):
            continue
        chrom = cols[0]
        start = int(cols[3]) - 1
        end = int(cols[4])
        strand = cols[6]
        attr = cols[8]
        m_gene = _GENE_NAME_RE.search(attr)
        m_tx = _TX_ID_RE.search(attr)
        if not m_gene or not m_tx:
            continue
        gene = m_gene.group(1)
        tx_id = m_tx.group(1)

        if feat == "transcript":
            is_canon = "tag \"Ensembl_canonical\"" in attr
            is_mane = "tag \"MANE_Select\"" in attr
            transcripts[tx_id] = {
                "gene": gene, "chrom": chrom, "strand": strand,
                "tx_start": start, "tx_end": end,
                "is_canon": is_canon, "is_mane": is_mane,
                "exons": [], "cds": [], "utr": [],
                "length": end - start,
            }
        else:
            d = transcripts.get(tx_id)
            if d is None:
                continue
            if feat == "exon":
                d["exons"].append((start, end))
            elif feat == "UTR":
                d["utr"].append((start, end))
            elif feat == "CDS":
                d["cds"].append((start, end))

    by_gene: dict[str, list] = {}
    for tx_id, d in transcripts.items():
        by_gene.setdefault(d["gene"], []).append(d)

    chosen = {}
    for gene, txs in by_gene.items():
        canon = [t for t in txs if t["is_canon"]]
        if not canon:
            canon = [t for t in txs if t["is_mane"]]
        if not canon:
            canon = txs
        canon = sorted(canon, key=lambda t: t["length"], reverse=True)
        chosen[gene] = canon[0]
    return chosen


def load_attribution_window(pert: str, gene: str, tss_pos: int,
                             view_start: int, view_end: int):
    h5_path = ATTR_DIR / pert / f"{pert}_union_raw_attribution.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as hf:
        if gene not in hf or "ixg" not in hf[gene] or "ixg_fc" not in hf[gene]:
            return None
        context_start = tss_pos - HALF_CONTEXT
        a_start = max(0, view_start - context_start)
        a_end = min(CONTEXT_LENGTH, view_end - context_start)
        if a_start >= a_end:
            return None
        ixg = hf[gene]["ixg"][a_start:a_end, :]
        ixg_fc = hf[gene]["ixg_fc"][a_start:a_end, :]
    pert_sum = ixg.sum(axis=1).astype(np.float64)
    diff_sum = ixg_fc.sum(axis=1).astype(np.float64)
    ctrl_sum = pert_sum - diff_sum
    coords = context_start + a_start + np.arange(a_end - a_start)
    return {
        "coords": coords,
        "pert": pert_sum,
        "ctrl": ctrl_sum,
        "diff": diff_sum,
    }


def bin_average(coords: np.ndarray, values: np.ndarray, bin_size: int = BIN_SIZE):
    n = (len(values) // bin_size) * bin_size
    if n == 0:
        return coords, values
    v = values[:n].reshape(-1, bin_size).mean(axis=1)
    c = coords[:n].reshape(-1, bin_size).mean(axis=1)
    return c, v


def load_logo_window(pert: str, gene: str, tss_pos: int,
                      win_start: int) -> np.ndarray | None:
    h5_path = ATTR_DIR / pert / f"{pert}_union_raw_attribution.h5"
    if not h5_path.exists():
        return None
    context_start = tss_pos - HALF_CONTEXT
    pos_start = win_start - context_start
    if pos_start < 0 or pos_start + WINDOW_LEN > CONTEXT_LENGTH:
        return None
    with h5py.File(h5_path, "r") as hf:
        if gene not in hf or "ixg_fc" not in hf[gene]:
            return None
        return hf[gene]["ixg_fc"][pos_start:pos_start + WINDOW_LEN, :]


_INSTANCE_RE = re.compile(
    r"(?P<pert>[^_]+(?:\.[^_]+)*)__"
    r"(?P<patdir>(?:pos|neg)_patterns)\.(?P<patname>pattern_\d+)__"
)


def _revcomp_pwm(arr: np.ndarray) -> np.ndarray:
    return arr[::-1, [3, 2, 1, 0]]


def load_modisco_pattern(pert: str, seqlet_instance_id: str,
                          is_revcomp: bool) -> np.ndarray | None:
    h5_path = MODISCO_DIR / pert / f"{pert}_modisco_v2.h5"
    if not h5_path.exists():
        return None
    m = _INSTANCE_RE.search(seqlet_instance_id)
    if not m:
        return None
    pat_dir = m.group("patdir")
    pat_name = m.group("patname")
    with h5py.File(h5_path, "r") as hf:
        if pat_dir not in hf or pat_name not in hf[pat_dir]:
            return None
        cs = np.array(hf[pat_dir][pat_name]["contrib_scores"])
    if is_revcomp:
        cs = _revcomp_pwm(cs)
    return cs


def draw_gene_model(ax, model: dict, view_start: int, view_end: int,
                    tss_pos: int, gene: str):
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if model is None:
        ax.text((view_start + view_end) / 2, 0,
                f"{gene}: no Gencode canonical transcript found",
                ha="center", va="center", fontsize=12, color="gray")
        return

    chrom = model["chrom"]
    strand = model["strand"]
    exons = sorted(model["exons"])

    if exons:
        bb_start = max(view_start, min(s for s, _ in exons))
        bb_end = min(view_end, max(e for _, e in exons))
        if bb_start < bb_end:
            ax.hlines(0, bb_start, bb_end, colors="black", linewidth=1.4,
                      zorder=1)
            step = max(1500, (view_end - view_start) // 25)
            for x in np.arange(bb_start + step / 2, bb_end, step):
                marker = ">" if strand == "+" else "<"
                ax.plot(x, 0, marker=marker, markersize=10,
                         color="black", zorder=2)

    for s, e in exons:
        if e <= view_start or s >= view_end:
            continue
        rs = max(s, view_start)
        re_ = min(e, view_end)
        rect = mpatches.Rectangle((rs, -0.40), re_ - rs, 0.80,
                                   facecolor="#a0a0a0", edgecolor="black",
                                   linewidth=1.0, zorder=3)
        ax.add_patch(rect)

    ax.axvline(tss_pos, color="black", linewidth=1.2, linestyle="--",
                alpha=0.85, zorder=6)
    ax.text(tss_pos, 0.55, "TSS", fontsize=BASE_FONT - 3, ha="center",
             va="bottom", fontweight="bold")

    ax.text(view_start + (view_end - view_start) * 0.005, 0.85,
             f"{gene} ({chrom} {strand})",
             fontsize=BASE_FONT, fontweight="bold", ha="left", va="bottom")


def draw_track(ax, coords: np.ndarray, values: np.ndarray, ymin: float, ymax: float,
                color_pos: str, color_neg: str, label: str,
                tss_pos: int,
                seqlet_start: int | None, seqlet_end: int | None,
                view_start: int, view_end: int, show_xticks: bool):
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(ymin, ymax)
    if (seqlet_start is not None and seqlet_end is not None
            and seqlet_end > view_start and seqlet_start < view_end):
        s = max(seqlet_start, view_start)
        e = min(seqlet_end, view_end)
        min_w = max(80, (view_end - view_start) // 60)
        if e - s < min_w:
            mid = (s + e) // 2
            half = min_w // 2
            s = max(view_start, mid - half)
            e = min(view_end, mid + half)
        ax.axvspan(s, e, alpha=0.30, color="gray", zorder=0)
    clipped = np.clip(values, ymin, ymax)
    ax.fill_between(coords, 0, clipped, where=clipped >= 0,
                     color=color_pos, alpha=0.85, linewidth=0, step="mid")
    ax.fill_between(coords, 0, clipped, where=clipped < 0,
                     color=color_neg, alpha=0.85, linewidth=0, step="mid")
    ax.axhline(0, color="gray", linewidth=0.4, zorder=0)
    ax.axvline(tss_pos, color="black", linewidth=0.8, linestyle="--",
                alpha=0.5, zorder=1)
    ax.set_ylabel(label, fontsize=BASE_FONT - 3, rotation=0, ha="right",
                   va="center", labelpad=14)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_xticks:
        flank = (view_end - view_start) // 2
        tick_step = max(5_000, flank // 5)
        first_tick = ((view_start + tick_step - 1) // tick_step) * tick_step
        ticks = np.arange(first_tick, view_end + 1, tick_step)
        ax.set_xticks(ticks)
        labels = [f"{int(t):,}" for t in ticks]
        ax.set_xticklabels(labels, fontsize=BASE_FONT - 6, rotation=30,
                            ha="right")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def draw_logo(ax, ixg_window: np.ndarray, win_start: int, win_end: int,
               core_start: int, core_end: int, core_label: str,
               chrom: str):
    df_logo = pd.DataFrame(ixg_window, columns=BASES, index=range(WINDOW_LEN))
    logomaker.Logo(df_logo, ax=ax, color_scheme=BASE_COLORS,
                    font_name="DejaVu Sans Mono")
    rs = max(0, core_start - win_start)
    re_ = min(WINDOW_LEN, core_end - win_start)
    if rs < re_:
        ax.axvspan(rs - 0.5, re_ - 0.5, alpha=0.20, color="gray", zorder=0)
        ax.text((rs + re_) / 2 - 0.5, ax.get_ylim()[1] * 0.95, core_label,
                 ha="center", va="top", fontsize=BASE_FONT - 3,
                 fontweight="bold", color="#cc3333")
    ax.set_xlim(-0.5, WINDOW_LEN - 0.5)
    tick_positions = list(range(0, WINDOW_LEN + 1, 16))
    tick_labels = [f"{win_start + p:,}" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=BASE_FONT - 7, rotation=30,
                        ha="right")
    ax.set_xlabel(
        f"Genomic position ({chrom}:{win_start:,}-{win_end:,})",
        fontsize=BASE_FONT - 4,
    )
    ax.set_ylabel("Differential\nAttribution",
                  fontsize=BASE_FONT - 4)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 6)


def draw_modisco_motif(ax, contrib_scores: np.ndarray | None,
                        motif_label: str, pattern_name: str,
                        is_revcomp: bool):
    if contrib_scores is None:
        ax.text(0.5, 0.5, "modisco pattern not found",
                ha="center", va="center", fontsize=BASE_FONT - 6,
                color="gray", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        return
    L = contrib_scores.shape[0]
    df_logo = pd.DataFrame(contrib_scores, columns=BASES, index=range(L))
    logomaker.Logo(df_logo, ax=ax, color_scheme=BASE_COLORS,
                    font_name="DejaVu Sans Mono")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_xticks([0, L // 2, L - 1])
    ax.set_xticklabels(["1", str(L // 2 + 1), str(L)],
                        fontsize=BASE_FONT - 7)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 7)
    ax.set_xlabel("pattern position", fontsize=BASE_FONT - 5)
    ax.set_ylabel("contrib_scores", fontsize=BASE_FONT - 5)
    revcomp_tag = " (rev-comp)" if is_revcomp else ""
    ax.set_title(
        f"TF-MoDISco cluster motif — {motif_label}{revcomp_tag}\n"
        f"({pattern_name})",
        fontsize=BASE_FONT - 7, pad=8,
    )


def draw_bar(ax, candidate_row):
    cats = ["WT ctrl", "WT pert", "Mut ctrl", "Mut pert"]
    vals = [candidate_row["wt_ctrl"], candidate_row["wt_pert"],
            candidate_row["mt_ctrl"], candidate_row["mt_pert"]]
    colors = [BAR_COLORS[c] for c in cats]
    xpos = np.arange(len(cats))
    ax.bar(xpos, vals, color=colors, edgecolor="black", linewidth=0.8,
            alpha=0.95)
    span = max(max(vals) - min(vals), 1e-6)
    for i, v in enumerate(vals):
        ax.text(i, v + span * 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=BASE_FONT - 7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats, fontsize=BASE_FONT - 4)
    ax.set_ylabel("Predicted log-expr.", fontsize=BASE_FONT - 4)
    pad = span * 0.20 + 0.02
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 6)
    ax.set_title(
        "Per-condition prediction",
        fontsize=BASE_FONT - 7, pad=8,
    )


def draw_fc_bar(ax, candidate_row):
    fc_wt = candidate_row["fc_wt"]
    fc_mt = candidate_row["fc_mt"]
    cats = ["WT", "Mut"]
    vals = [fc_wt, fc_mt]
    colors = [BAR_COLORS["WT pert"], BAR_COLORS["Mut pert"]]
    xpos = np.arange(len(cats))
    ax.bar(xpos, vals, color=colors, edgecolor="black", linewidth=0.8,
            alpha=0.95, width=0.65)
    span = max(abs(fc_wt), abs(fc_mt), 1e-6)
    for i, v in enumerate(vals):
        ax.text(i, v + np.sign(v if v != 0 else 1) * span * 0.05,
                 f"{v:+.2f}", ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=BASE_FONT - 7)
    ax.axhline(0, color="black", linewidth=0.6, zorder=0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats, fontsize=BASE_FONT - 4)
    ax.set_ylabel("FC = pert − ctrl", fontsize=BASE_FONT - 4)
    pad = span * 0.30 + 0.02
    ax.set_ylim(-span - pad if min(vals) < 0 else -pad,
                 +span + pad if max(vals) > 0 else +pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=BASE_FONT - 6)
    cancel_pct = (abs(fc_wt) - abs(fc_mt)) / max(abs(fc_wt), 1e-6) * 100
    ax.set_title(
        f"FC$_{{WT}}$ vs FC$_{{MT}}$\ncancel={cancel_pct:.0f}%",
        fontsize=BASE_FONT - 7, pad=8,
    )


def make_figure(row, gene_models, tss_map, train_tss_map,
                 view_flank: int | None, min_flank: int, margin: int,
                 atac_state):
    gene = row["gene"]
    pert = row["perturbation"]
    chrom = row["chromosome"]

    if gene not in train_tss_map:
        return None, "no training BED TSS"
    bed_chrom, bed_tss, bed_strand = train_tss_map[gene]
    tss_pos = bed_tss
    if bed_chrom != chrom:
        return None, f"chrom mismatch {bed_chrom}!={chrom}"

    core_mid_pre = (int(row["core_genomic_start"]) +
                     int(row["core_genomic_end"])) // 2
    view_start, view_end = _compute_view_window(
        tss_pos, core_mid_pre, view_flank, min_flank, margin,
    )

    attr = load_attribution_window(pert, gene, tss_pos, view_start, view_end)
    if attr is None:
        return None, "no attribution"

    pert_c, pert_v = bin_average(attr["coords"], attr["pert"])
    ctrl_c, ctrl_v = bin_average(attr["coords"], attr["ctrl"])
    diff_c, diff_v = bin_average(attr["coords"], attr["diff"])

    pc_max = max(np.abs(pert_v).max(), np.abs(ctrl_v).max())
    pc_max = max(pc_max, 1e-6)
    diff_max = max(np.abs(diff_v).max(), 1e-6)

    atac_pert = atac_utils.window_track(
        atac_state["matrix"], atac_state["by_chrom"], atac_state["columns"],
        chrom, view_start, view_end, pert, bin_size=BIN_SIZE,
    )
    atac_ctrl = atac_utils.window_track(
        atac_state["matrix"], atac_state["by_chrom"], atac_state["columns"],
        chrom, view_start, view_end, ATAC_CTRL_COL, bin_size=BIN_SIZE,
    )
    atac_diff = atac_pert - atac_ctrl
    atac_coords = atac_utils.window_coords(view_start, view_end, BIN_SIZE)
    atac_pc_max = max(float(np.max(atac_pert)), float(np.max(atac_ctrl)), 1e-6)
    atac_diff_max = max(float(np.abs(atac_diff).max()), 1e-6)

    core_mid = (int(row["core_genomic_start"]) + int(row["core_genomic_end"])) // 2
    win_start = core_mid - WINDOW_LEN // 2
    win_end = win_start + WINDOW_LEN
    logo_arr = load_logo_window(pert, gene, tss_pos, win_start)
    if logo_arr is None:
        return None, "no logo window"

    core_start = int(row["core_genomic_start"])
    core_end = int(row["core_genomic_end"])

    fig = plt.figure(figsize=(22, 23.0))
    outer = fig.add_gridspec(
        nrows=3, ncols=1, height_ratios=[4.8, 1.3, 2.2], hspace=0.55,
    )
    gs_top = outer[0].subgridspec(
        nrows=7, ncols=1,
        height_ratios=[0.85, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95], hspace=0.20,
    )
    gs_bot = outer[2].subgridspec(
        nrows=1, ncols=4,
        width_ratios=[1.0, 0.55, 1.0, 0.45],
        wspace=0.45,
    )

    ax_gene = fig.add_subplot(gs_top[0])
    draw_gene_model(ax_gene, gene_models.get(gene), view_start, view_end,
                     tss_pos, gene)

    pert_label = pert.replace("MartinRufino.", "")

    ax_atac_pert = fig.add_subplot(gs_top[1], sharex=ax_gene)
    ax_atac_ctrl = fig.add_subplot(gs_top[2], sharex=ax_gene)
    ax_atac_diff = fig.add_subplot(gs_top[3], sharex=ax_gene)
    draw_track(ax_atac_pert, atac_coords, atac_pert, 0.0, atac_pc_max * 1.10,
                ATAC_POS_COLOR, ATAC_POS_COLOR,
                f"{pert_label}\nATAC-seq", tss_pos, core_start, core_end,
                view_start, view_end, show_xticks=False)
    draw_track(ax_atac_ctrl, atac_coords, atac_ctrl, 0.0, atac_pc_max * 1.10,
                ATAC_POS_COLOR, ATAC_POS_COLOR,
                "Ctrl\nATAC-seq", tss_pos, core_start, core_end,
                view_start, view_end, show_xticks=False)
    draw_track(ax_atac_diff, atac_coords, atac_diff,
                -atac_diff_max * 1.10, atac_diff_max * 1.10,
                DIFF_POS_COLOR, DIFF_NEG_COLOR,
                f"{pert_label} − Ctrl\nATAC-seq", tss_pos, core_start, core_end,
                view_start, view_end, show_xticks=False)

    ax_pert = fig.add_subplot(gs_top[4], sharex=ax_gene)
    ax_ctrl = fig.add_subplot(gs_top[5], sharex=ax_gene)
    ax_diff = fig.add_subplot(gs_top[6], sharex=ax_gene)
    draw_track(ax_pert, pert_c, pert_v, -pc_max * 1.05, pc_max * 1.05,
                DIFF_POS_COLOR, DIFF_NEG_COLOR,
                f"{pert_label}\nAttribution",
                tss_pos, core_start, core_end, view_start, view_end, show_xticks=False)
    draw_track(ax_ctrl, ctrl_c, ctrl_v, -pc_max * 1.05, pc_max * 1.05,
                DIFF_POS_COLOR, DIFF_NEG_COLOR, "Ctrl\nAttribution",
                tss_pos, core_start, core_end, view_start, view_end, show_xticks=False)
    draw_track(ax_diff, diff_c, diff_v, -diff_max * 1.05, diff_max * 1.05,
                DIFF_POS_COLOR, DIFF_NEG_COLOR,
                f"{pert_label} − Ctrl\nAttribution",
                tss_pos, core_start, core_end, view_start, view_end, show_xticks=True)

    ax_logo = fig.add_subplot(outer[1])
    draw_logo(ax_logo, logo_arr, win_start, win_end,
              core_start, core_end, core_label=pert_label, chrom=chrom)
    motif_label = str(row["matched_motif_gene"])

    ax_bar = fig.add_subplot(gs_bot[0])
    draw_bar(ax_bar, row)

    ax_fc_bar = fig.add_subplot(gs_bot[1])
    draw_fc_bar(ax_fc_bar, row)

    ax_modisco = fig.add_subplot(gs_bot[2])
    seqlet_id = str(row.get("seqlet_instance_id", ""))
    is_revcomp = bool(row.get("is_revcomp", False))
    pattern_arr = load_modisco_pattern(pert, seqlet_id, is_revcomp)
    pattern_match = _INSTANCE_RE.search(seqlet_id)
    pattern_label = (
        f"{pattern_match.group('patdir')}/{pattern_match.group('patname')}"
        if pattern_match else "?"
    )
    draw_modisco_motif(ax_modisco, pattern_arr, motif_label,
                       pattern_label, is_revcomp)

    title = (
        f"{gene} | perturbation: {pert_label} | seqlet motif match: "
        f"{motif_label}  |  attribution = {row['attr_sum_abs']:.3f}  |  "
        f"|FC_WT|={abs(row['fc_wt']):.2f}, cancel={row['cancel']:.2f}"
    )
    fig.suptitle(title, fontsize=BASE_FONT + 2, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.05)
    return fig, None


def main():
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TSS maps...")
    tss_map = load_tss_bed()
    train_tss_map = load_training_bed_tss()

    print("Loading candidates...")
    cand = pd.read_csv(CANDIDATES_TSV, sep="\t")
    if args.gene_filter:
        cand = cand[cand["gene"] == args.gene_filter]
    if args.limit:
        cand = cand.head(args.limit)
    print(f"  {len(cand)} candidates to render")

    print("Pre-extracting Gencode entries for candidate genes...")
    raw_text = grep_gtf_for_genes(cand["gene"].tolist())
    gene_models = parse_gene_models(raw_text)
    print(f"  {len(gene_models)} canonical transcripts loaded "
          f"(asked for {cand['gene'].nunique()} genes)")

    print("Loading ATAC pseudobulk CPM table...")
    atac_columns, atac_matrix, atac_by_chrom = atac_utils.load_atac_table(ATAC_TSV)
    atac_state = {"columns": atac_columns, "matrix": atac_matrix,
                  "by_chrom": atac_by_chrom}
    print(f"  {atac_matrix.shape[0]:,} peaks × {atac_matrix.shape[1]} columns")

    summary_rows = []
    n_ok = 0
    for _, row in cand.iterrows():
        fig, err = make_figure(row, gene_models, tss_map, train_tss_map,
                                view_flank=args.flank,
                                min_flank=args.min_flank,
                                margin=args.margin,
                                atac_state=atac_state)
        gene = row["gene"]
        pert = row["perturbation"].replace("MartinRufino.", "")
        if fig is None:
            print(f"  SKIP {gene}/{pert}: {err}")
            continue
        out_name = f"{gene}__{pert}__{row['chromosome']}_{int(row['core_genomic_start'])}"
        svg = FIG_DIR / f"{out_name}.svg"
        png = FIG_DIR / f"{out_name}.png"
        fig.savefig(svg, format="svg", bbox_inches="tight")
        fig.savefig(png, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        n_ok += 1
        summary_rows.append({
            "gene": gene, "perturbation": row["perturbation"],
            "chromosome": row["chromosome"],
            "core_start": int(row["core_genomic_start"]),
            "core_end": int(row["core_genomic_end"]),
            "matched_motif_gene": row["matched_motif_gene"],
            "attr_sum_abs": row["attr_sum_abs"],
            "fc_wt": row["fc_wt"], "fc_mt": row["fc_mt"],
            "cancel": row["cancel"],
            "atac_max_abs_delta": row.get("atac_max_abs_delta"),
            "atac_signed_delta": row.get("atac_signed_delta"),
            "svg": str(svg.relative_to(CWD)),
        })
        if n_ok % 5 == 0:
            print(f"  {n_ok} figures rendered")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DATA_DIR / "figure_summary.tsv", sep="\t", index=False)
    print(f"\nDone: {n_ok}/{len(cand)} figures -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
