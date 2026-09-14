#conda activate singlecell
"""Quantify how separable perturbations are in the *observed* data.

Fig. 2e reports the agreement (ARI / NMI / FM) between leiden clusters derived
from observed and from fitted expression profiles. That agreement is bounded by
how much separable perturbation structure exists in the observed data in the
first place: if every perturbation response looks like every other one, the
cluster boundaries are essentially arbitrary and no model can reproduce them.

This script measures that bound directly from the observed pseudobulk matrix,
without leiden and without any cluster labels:

  1. control-subtracted response profiles  delta = X_pert - X_control
     (values are log-normalised expression, so this is a log fold change)
  2. top-variance genes are kept so that unresponsive genes do not dilute the
     correlation structure
  3. pairwise Pearson correlation between perturbation profiles, from which two
     quantities are taken per perturbation
       nearest-neighbour correlation : max correlation with any other perturbation
       background correlation       : median correlation with all other perturbations
  4. gap = nearest - background, summarised by its median over perturbations

Interpretation of the gap:
  large  -> perturbations form groups (related perturbations correlate with each
            other but not with unrelated ones)
  small  -> no group structure, either because responses are noise-dominated
            (nearest ~ background ~ 0) or because a single dominant axis makes
            every perturbation resemble every other one (nearest ~ background ~ high)
The difference is what makes this robust to both failure modes; the raw mean
correlation alone would only detect the first.

Two controls are computed alongside:
  * permutation null - genes are shuffled independently within each perturbation
    profile. This preserves each perturbation's effect-size distribution but
    destroys the gene-identity correspondence between perturbations, so it also
    absorbs the upward bias that the max statistic has when a dataset has many
    perturbations.
  * subsampling      - the gap is recomputed on a fixed number of perturbations
    so that datasets with very different perturbation counts can be compared
    directly.

Outputs (all under across_study/perturbation_separation/):
  perturbation_separation_metrics.txt        one row per study
  perturbation_separation_per_perturbation.txt  nearest / background / gap per perturbation
  gap_barplot.svg                            observed vs permuted gap per study
  gap_ratio_barplot.svg                      gap / permuted null per study
  nn_bg_distribution.svg                     nearest vs background distributions
  corr_clustermap_{study_name}.svg           perturbation x perturbation correlation
  gap_vs_agreement_{model}.svg               gap vs Fig. 2e ARI / NMI

Environment variables:
  TARGET_STUDIES      comma-separated study names to restrict the run
  N_TOP_GENES         number of top-variance genes (default 2000)
  N_PERM              permutation-null repeats (default 20)
  SUBSAMPLE_N         perturbations per subsample (default 100)
  N_SUBSAMPLE         subsampling repeats (default 50)
  MAX_HEATMAP_PERTS   cap on perturbations drawn in the clustermap (default 1500)
  RANDOM_SEED         seed (default 0)
  FORCE_REGENERATE    recompute even if the metrics table already exists
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from dataset_model_config import (
    all_studies,
    all_datanames,
    studies as active_studies,
    datanames as active_datanames,
    pretrained_models,
    study_suffixes,
    model_fixed_suffix,
    name_replace,
)


OUT_DIR = "across_study/perturbation_separation"

EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"
OBSERVED_COLOR = "#BC5765"
NULL_COLOR = "#B5BBC1"
NN_COLOR = "#BC5765"
BG_COLOR = "#8E949A"

PAPER_FONT_SIZE = 7

N_TOP_GENES = int(os.environ.get("N_TOP_GENES", 2000))
N_PERM = int(os.environ.get("N_PERM", 20))
SUBSAMPLE_N = int(os.environ.get("SUBSAMPLE_N", 100))
N_SUBSAMPLE = int(os.environ.get("N_SUBSAMPLE", 50))
MAX_HEATMAP_PERTS = int(os.environ.get("MAX_HEATMAP_PERTS", 1500))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 0))

# Block size for the pairwise-correlation scan. The full perturbation x
# perturbation matrix is never materialised, so genome-wide Replogle
# (~10^4 perturbations) stays within a few hundred MB.
CORR_BLOCK = 512


def _apply_paper_rcparams(font_size=PAPER_FONT_SIZE):
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["legend.title_fontsize"] = font_size
    plt.rcParams["axes.edgecolor"] = EDGE_COLOR
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["xtick.color"] = EDGE_COLOR
    plt.rcParams["ytick.color"] = EDGE_COLOR
    plt.rcParams["xtick.labelcolor"] = "black"
    plt.rcParams["ytick.labelcolor"] = "black"
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6


def resolve_studies():
    target = os.environ.get("TARGET_STUDIES")
    if not target:
        return list(active_studies), list(active_datanames)
    names = [s.strip() for s in target.split(",") if s.strip()]
    out_studies, out_datanames = [], []
    for name in names:
        if name not in all_studies:
            print(f"[warn] unknown study: {name}; skipping")
            continue
        idx = all_studies.index(name)
        out_studies.append(all_studies[idx])
        out_datanames.append(all_datanames[idx])
    return out_studies, out_datanames


def load_response_matrix(study_name):
    """Return (delta matrix as perturbation x gene DataFrame, control column name).

    data/{study_name}.tsv is gene x sample with the first column holding the
    train/valid/test split and the first value column holding the control
    channel (same convention as 21_embedding_signature.py and
    scripts/attribution_analysis/01-generate_tasks.py).
    """
    path = f"data/{study_name}.tsv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, sep="\t", index_col=0)
    if "training" in df.columns:
        df = df.drop("training", axis=1)
    ctrl_col = df.columns[0]
    pert_cols = df.columns[1:]
    if len(pert_cols) < 3:
        print(f"[skip] {study_name}: only {len(pert_cols)} perturbations")
        return None, None
    delta = df[pert_cols].sub(df[ctrl_col], axis=0).T
    delta.index.name = "perturbation"
    return delta, ctrl_col


def select_top_variance_genes(delta, n_top=N_TOP_GENES):
    variances = delta.var(axis=0, ddof=1)
    variances = variances.replace([np.inf, -np.inf], np.nan).dropna()
    keep = variances.sort_values(ascending=False).index[:n_top]
    # Preserve the original gene order so that the matrix is reproducible.
    keep = [g for g in delta.columns if g in set(keep)]
    return delta[keep]


def standardize_rows(matrix):
    """Row-centre and row-scale so that Z @ Z.T / n_genes is Pearson correlation."""
    values = np.asarray(matrix, dtype=np.float32)
    centred = values - values.mean(axis=1, keepdims=True)
    sd = centred.std(axis=1, ddof=0, keepdims=True)
    ok = sd[:, 0] > 0
    sd = np.where(sd == 0, 1.0, sd)
    return centred / sd, ok


def nearest_background(z, block=CORR_BLOCK):
    """Per-row nearest-neighbour and background correlation.

    Computed blockwise so the full pairwise matrix is never held in memory.
    The self-correlation is excluded from both statistics.
    """
    n_perts, n_genes = z.shape
    nearest = np.empty(n_perts, dtype=np.float64)
    background = np.empty(n_perts, dtype=np.float64)
    for start in range(0, n_perts, block):
        stop = min(start + block, n_perts)
        corr = (z[start:stop] @ z.T) / np.float32(n_genes)
        corr = corr.astype(np.float64)
        rows = np.arange(stop - start)
        corr[rows, np.arange(start, stop)] = np.nan
        nearest[start:stop] = np.nanmax(corr, axis=1)
        background[start:stop] = np.nanmedian(corr, axis=1)
    return nearest, background


def gap_summary(nearest, background):
    gap = nearest - background
    return {
        "nn_median": float(np.median(nearest)),
        "bg_median": float(np.median(background)),
        "bg_mean": float(np.mean(background)),
        "gap_median": float(np.median(gap)),
        "gap_q25": float(np.percentile(gap, 25)),
        "gap_q75": float(np.percentile(gap, 75)),
        "gap_mean": float(np.mean(gap)),
    }


def permutation_null(z, n_perm=N_PERM, seed=RANDOM_SEED):
    """Gap expected when gene identity carries no shared structure.

    Shuffling genes independently within each row keeps every perturbation's
    effect-size distribution intact and only destroys the correspondence
    between perturbations, so the resulting gap is the floor set by the number
    of perturbations, the number of genes, and the noise level.
    """
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_perm):
        z_perm = rng.permuted(z, axis=1)
        nearest, background = nearest_background(z_perm)
        gaps.append(float(np.median(nearest - background)))
    gaps = np.asarray(gaps)
    return {
        "null_gap_median": float(np.median(gaps)),
        "null_gap_mean": float(np.mean(gaps)),
        "null_gap_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else np.nan,
    }


def subsampled_gap(z, n_sub=SUBSAMPLE_N, n_repeat=N_SUBSAMPLE, seed=RANDOM_SEED):
    """Gap at a fixed perturbation count, so datasets of different size compare."""
    n_perts = z.shape[0]
    if n_perts < n_sub:
        return {"gap_subsampled_median": np.nan, "gap_subsampled_sd": np.nan}
    rng = np.random.default_rng(seed + 1)
    gaps = []
    for _ in range(n_repeat):
        idx = rng.choice(n_perts, size=n_sub, replace=False)
        nearest, background = nearest_background(z[idx])
        gaps.append(float(np.median(nearest - background)))
    gaps = np.asarray(gaps)
    return {
        "gap_subsampled_median": float(np.median(gaps)),
        "gap_subsampled_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else np.nan,
    }


def analyse_study(study_name, dataname):
    delta, ctrl_col = load_response_matrix(study_name)
    if delta is None:
        return None, None, None

    delta = select_top_variance_genes(delta)
    z, ok = standardize_rows(delta)
    if (~ok).any():
        dropped = int((~ok).sum())
        print(f"[warn] {study_name}: dropping {dropped} perturbations with zero variance")
        z = z[ok]
        delta = delta.loc[ok]
    if z.shape[0] < 3:
        print(f"[skip] {study_name}: fewer than 3 usable perturbations")
        return None, None, None

    nearest, background = nearest_background(z)

    record = {
        "Study": dataname,
        "study_name": study_name,
        "control_column": ctrl_col,
        "n_perturbations": int(z.shape[0]),
        "n_genes_used": int(z.shape[1]),
    }
    record.update(gap_summary(nearest, background))
    record.update(permutation_null(z))
    record.update(subsampled_gap(z))
    null_median = record["null_gap_median"]
    null_sd = record["null_gap_sd"]
    record["gap_ratio"] = (
        record["gap_median"] / null_median if null_median and null_median > 0 else np.nan
    )
    record["gap_z"] = (
        (record["gap_median"] - record["null_gap_mean"]) / null_sd
        if null_sd and null_sd > 0
        else np.nan
    )

    per_pert = pd.DataFrame(
        {
            "Study": dataname,
            "study_name": study_name,
            "perturbation": delta.index,
            "nearest_correlation": nearest,
            "background_correlation": background,
            "gap": nearest - background,
        }
    )
    return record, per_pert, (z, list(delta.index))


def plot_corr_clustermap(z, pert_names, study_name, dataname, seed=RANDOM_SEED):
    """Perturbation x perturbation correlation, ordered by hierarchical clustering."""
    n_perts = z.shape[0]
    if n_perts > MAX_HEATMAP_PERTS:
        rng = np.random.default_rng(seed + 2)
        idx = np.sort(rng.choice(n_perts, size=MAX_HEATMAP_PERTS, replace=False))
        z = z[idx]
        pert_names = [pert_names[i] for i in idx]
        subtitle = f"{MAX_HEATMAP_PERTS} of {n_perts} perturbations"
    else:
        subtitle = f"{n_perts} perturbations"

    corr = np.clip((z @ z.T) / np.float32(z.shape[1]), -1.0, 1.0).astype(np.float64)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=pert_names, columns=pert_names)

    _apply_paper_rcparams()
    grid = sns.clustermap(
        corr_df,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        figsize=(7.5 / 2.54, 7.5 / 2.54),
        dendrogram_ratio=0.12,
        cbar_pos=(0.02, 0.82, 0.03, 0.14),
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.figure.suptitle(f"{dataname}\n{subtitle}", fontsize=PAPER_FONT_SIZE)
    grid.figure.savefig(
        f"{OUT_DIR}/corr_clustermap_{study_name}.svg", dpi=300, bbox_inches="tight"
    )
    plt.close(grid.figure)


def plot_gap_barplot(metrics, study_order):
    long = []
    for _, row in metrics.iterrows():
        long.append({"Study": row["Study"], "Source": "Observed", "gap": row["gap_median"]})
        long.append(
            {"Study": row["Study"], "Source": "Permuted null", "gap": row["null_gap_median"]}
        )
    long = pd.DataFrame(long)

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(13 / 2.54, 6 / 2.54), dpi=300)
    sns.barplot(
        data=long,
        x="Study",
        y="gap",
        hue="Source",
        order=study_order,
        hue_order=["Observed", "Permuted null"],
        palette={"Observed": OBSERVED_COLOR, "Permuted null": NULL_COLOR},
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        ax=ax,
    )
    ax.set_ylabel("Nearest-neighbour gap\n(median over perturbations)")
    ax.set_xlabel("Study")
    ax.set_title("Separation of observed perturbation responses")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/gap_barplot.svg")
    plt.close(fig)


def plot_gap_ratio_barplot(metrics, study_order, sort_by_ratio=False):
    """Gap expressed relative to the permutation null computed for the same study.

    The raw gap is not comparable between studies. The nearest-neighbour term is
    a maximum over the remaining perturbations, so it grows with the number of
    perturbations; a study with few perturbations has both a smaller gap and a
    smaller null. The permutation null is computed at that study's own
    perturbation count, so the ratio removes the dependence and is the quantity
    that can be read across studies. A ratio of 1 means the observed separation
    is indistinguishable from the no-shared-structure floor.

    Absolute gap and null values are deliberately left to the source-data table
    rather than annotated here; this panel is about the normalised comparison.

    Studies keep `study_order`, which is the order used by the other figures, so
    that panels can be read against each other. `sort_by_ratio` is available for
    inspection but should stay off for anything that goes into the manuscript.
    """
    table = metrics.dropna(subset=["gap_ratio"]).copy()
    table = table[table["Study"].isin(study_order)]
    if table.empty:
        print("[skip] gap ratio barplot: no study has a usable null")
        return
    if sort_by_ratio:
        table = table.sort_values("gap_ratio", ascending=True)
    else:
        rank = {name: i for i, name in enumerate(study_order)}
        table = table.sort_values("Study", key=lambda s: s.map(rank), ascending=False)

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(10.5 / 2.54, 6.8 / 2.54), dpi=300)
    y = np.arange(len(table))
    ax.barh(
        y,
        table["gap_ratio"],
        height=0.68,
        color=OBSERVED_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        zorder=3,
    )
    ax.axvline(1, color=LINE_COLOR, linewidth=0.7, linestyle="--", zorder=4)
    ax.text(
        1.09,
        len(table) - 0.32,
        "permutation null",
        fontsize=PAPER_FONT_SIZE - 1,
        color=LINE_COLOR,
        va="bottom",
        ha="left",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["Study"])
    ax.set_ylim(-0.65, len(table) + 0.15)
    ax.set_xlim(0, float(table["gap_ratio"].max()) * 1.08)
    ax.set_xlabel("Nearest-neighbour gap relative to permutation null\n(observed / null)")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/gap_ratio_barplot.svg")
    plt.close(fig)


def plot_nn_bg_distribution(per_pert, study_order):
    long = per_pert.melt(
        id_vars=["Study"],
        value_vars=["nearest_correlation", "background_correlation"],
        var_name="Statistic",
        value_name="correlation",
    )
    long["Statistic"] = long["Statistic"].map(
        {
            "nearest_correlation": "Nearest neighbour",
            "background_correlation": "Background (median)",
        }
    )

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(14 / 2.54, 6 / 2.54), dpi=300)
    sns.violinplot(
        data=long,
        x="Study",
        y="correlation",
        hue="Statistic",
        order=study_order,
        hue_order=["Nearest neighbour", "Background (median)"],
        palette={"Nearest neighbour": NN_COLOR, "Background (median)": BG_COLOR},
        split=True,
        inner="quartile",
        linewidth=0.4,
        cut=0,
        ax=ax,
    )
    ax.axhline(0, color=LINE_COLOR, linewidth=0.4, linestyle="--")
    ax.set_ylabel("Correlation between\nperturbation responses")
    ax.set_xlabel("Study")
    ax.set_title("Nearest-neighbour vs background similarity (observed)")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/nn_bg_distribution.svg")
    plt.close(fig)


def load_clustering_agreement(study_name, pretrained_model):
    """Fig. 2e agreement for one study x model, if 21/22 have already been run."""
    suffixes = (
        [model_fixed_suffix[pretrained_model]]
        if pretrained_model in model_fixed_suffix
        else list(study_suffixes)
    )
    for suffix in suffixes:
        study = f"{study_name}__{pretrained_model}_{suffix}"
        path = f"figures/{study}/embedding/clustering_metrics.txt"
        if not os.path.exists(path):
            continue
        table = pd.read_csv(path, sep="\t")
        table = table.query('Comparison == "real_all - pred_all"')
        out = {}
        for metric in ["ARI", "NMI"]:
            hit = table.query("metrics == @metric")
            if not hit.empty:
                out[metric] = float(hit["value"].iloc[0])
        if out:
            out["study_dir"] = study
            return out
    return None


def plot_gap_vs_agreement(metrics, pretrained_model):
    rows = []
    for _, row in metrics.iterrows():
        agreement = load_clustering_agreement(row["study_name"], pretrained_model)
        if agreement is None:
            continue
        rows.append(
            {
                "Study": row["Study"],
                "gap_median": row["gap_median"],
                "ARI": agreement.get("ARI", np.nan),
                "NMI": agreement.get("NMI", np.nan),
            }
        )
    if len(rows) < 3:
        print(f"[skip] gap-vs-agreement for {pretrained_model}: only {len(rows)} studies")
        return None
    joined = pd.DataFrame(rows)

    model_label = name_replace.get(pretrained_model, pretrained_model)
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(13 / 2.54, 6 / 2.54), dpi=300)
    for ax, metric in zip(axes, ["ARI", "NMI"]):
        sub = joined.dropna(subset=[metric])
        ax.scatter(
            sub["gap_median"],
            sub[metric],
            s=14,
            color=OBSERVED_COLOR,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        for _, point in sub.iterrows():
            ax.annotate(
                point["Study"],
                (point["gap_median"], point[metric]),
                textcoords="offset points",
                xytext=(3, 2),
                fontsize=4.5,
                color="black",
            )
        if len(sub) >= 3:
            rho, pval = spearmanr(sub["gap_median"], sub[metric])
            ax.set_title(f"{metric}  (Spearman r = {rho:.2f}, P = {pval:.1e})")
        else:
            ax.set_title(metric)
        ax.set_xlabel("Nearest-neighbour gap (observed)")
        ax.set_ylabel(f"{metric} (observed vs fitted leiden)")
        sns.despine(ax=ax)
    fig.suptitle(model_label, fontsize=PAPER_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/gap_vs_agreement_{pretrained_model}.svg")
    plt.close(fig)
    return joined


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics_path = f"{OUT_DIR}/perturbation_separation_metrics.txt"
    per_pert_path = f"{OUT_DIR}/perturbation_separation_per_perturbation.txt"

    target_studies, target_datanames = resolve_studies()

    if os.path.exists(metrics_path) and not os.environ.get("FORCE_REGENERATE"):
        print(f"[skip] {metrics_path} exists; set FORCE_REGENERATE=1 to recompute")
        metrics = pd.read_csv(metrics_path, sep="\t")
        per_pert = pd.read_csv(per_pert_path, sep="\t")
    else:
        records, per_perts = [], []
        for study_name, dataname in zip(target_studies, target_datanames):
            print(f"[run] {study_name}")
            record, per_pert, matrix = analyse_study(study_name, dataname)
            if record is None:
                continue
            records.append(record)
            per_perts.append(per_pert)
            z, pert_names = matrix
            plot_corr_clustermap(z, pert_names, study_name, dataname)
            print(
                f"      n_pert={record['n_perturbations']} "
                f"gap={record['gap_median']:.3f} "
                f"null={record['null_gap_median']:.3f} "
                f"ratio={record['gap_ratio']:.2f}"
            )
        if not records:
            print("[error] no study produced metrics")
            return
        metrics = pd.DataFrame(records)
        per_pert = pd.concat(per_perts, ignore_index=True)
        metrics.to_csv(metrics_path, sep="\t", index=False)
        per_pert.to_csv(per_pert_path, sep="\t", index=False)

    study_order = [d for d in target_datanames if d in set(metrics["Study"])]
    plot_gap_barplot(metrics, study_order)
    plot_gap_ratio_barplot(metrics, study_order)
    plot_nn_bg_distribution(per_pert, study_order)
    for pretrained_model in pretrained_models:
        joined = plot_gap_vs_agreement(metrics, pretrained_model)
        if joined is not None:
            joined.to_csv(
                f"{OUT_DIR}/gap_vs_agreement_{pretrained_model}.txt", sep="\t", index=False
            )

    print(f"[done] wrote {metrics_path}")


if __name__ == "__main__":
    main()
