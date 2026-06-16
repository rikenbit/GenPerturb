#!/usr/bin/env python
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import wilcoxon



SCORE_META = {
    'attr_top10_score':    {'label': 'GenPerturb',   'color': '#1f77b4'},
    're2g_ext_score':      {'label': 'rE2G ext',     'color': '#9467bd'},
    're2g_score':          {'label': 'rE2G',         'color': '#2ca02c'},
    'abc_score':           {'label': 'ABC',          'color': '#ff7f0e'},
    'tss_distance_score':  {'label': 'TSS distance', 'color': '#7f7f7f'},
}

SCORE_COLUMNS = [
    'attr_top10_score',
    're2g_ext_score',
    're2g_score',
    'abc_score',
    'tss_distance_score',
]


def get_label(c: str) -> str:
    return SCORE_META.get(c, {}).get('label', c)


def get_color(c: str) -> str:
    return SCORE_META.get(c, {}).get('color', '#333333')


plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})



def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, None, None
    if np.all(y_score == 0):
        return np.nan, None, None
    auprc = float(average_precision_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auprc, precision, recall


def bootstrap_auprc_ci(y_true: np.ndarray, y_score: np.ndarray,
                        n_bootstrap: int = 1000, seed: int = 42,
                        ci_level: float = 0.95):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    if np.all(y_score == 0):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2 or np.all(ys == 0):
            boot_vals[i] = np.nan
            continue
        boot_vals[i] = float(average_precision_score(yt, ys))
    boot_vals = boot_vals[~np.isnan(boot_vals)]
    if len(boot_vals) < 10:
        return np.nan, np.nan
    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(boot_vals, 100 * alpha))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    return lo, hi


def evaluate_scored_peaks(scored_dir: str, degs_by_pert: Dict[str, set],
                           score_cols: List[str]):
    rows_counts = []
    rows_metrics = []
    pr_data = []  # (pert, {sc: (precision, recall, auprc)})
    filtered_dfs = []  # (pert, df_filt) for downstream stratified analysis

    files = sorted(f for f in os.listdir(scored_dir)
                   if f.startswith('scored_peaks_') and f.endswith('.tsv')
                   and f != 'scored_peaks_summary.tsv')
    if not files:
        raise FileNotFoundError(f'[ERROR] No scored_peaks files in {scored_dir}')

    print(f'[INFO] Found {len(files)} scored_peaks files in {scored_dir}')

    for fname in files:
        pert = fname[len('scored_peaks_'):-len('.tsv')]
        tf = pert.split('.')[-1]
        path = os.path.join(scored_dir, fname)
        df = pd.read_csv(path, sep='\t')
        n_total = len(df)
        n_genes_total = df['gene'].nunique() if 'gene' in df.columns else 0
        if 'atac_overlap' not in df.columns:
            print(f'[WARN] {pert}: no atac_overlap column. skip.')
            continue

        deg_set = degs_by_pert.get(tf, set())
        n_degs = len(deg_set)

        if n_degs == 0:
            print(f'[WARN] {pert} (TF={tf}): no DEGs found in source. skip.')
            continue

        if 'gene' not in df.columns:
            print(f'[WARN] {pert}: no gene column. skip.')
            continue

        df_filt = df[df['gene'].astype(str).isin(deg_set)].copy()
        n_genes_after = df_filt['gene'].nunique()
        n_peaks_after = len(df_filt)
        n_pos = int(df_filt['atac_overlap'].sum()) if n_peaks_after else 0
        n_neg = n_peaks_after - n_pos
        positive_rate = (n_pos / n_peaks_after) if n_peaks_after else 0.0

        rows_counts.append({
            'pert': pert,
            'tf_symbol': tf,
            'n_peaks_total': n_total,
            'n_genes_in_scored_peaks': n_genes_total,
            'n_degs_total': n_degs,
            'n_genes_after_deg_filter': n_genes_after,
            'n_peaks_after_deg_filter': n_peaks_after,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'positive_rate': positive_rate,
            'pos_neg_ratio': (n_pos / n_neg) if n_neg > 0 else np.nan,
        })

        if n_pos == 0 or n_neg == 0:
            print(f'[WARN] {pert}: after DEG filter n_pos={n_pos}, n_neg={n_neg}. '
                  f'AUPRC undefined; recorded counts only.')
            continue

        y_true = df_filt['atac_overlap'].astype(int).values
        per_score = {}
        metric_row = {
            'pert': pert,
            'tf_symbol': tf,
            'n_peaks_after_deg_filter': n_peaks_after,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'positive_rate': positive_rate,
        }
        for sc in score_cols:
            if sc not in df_filt.columns:
                metric_row[f'auprc_{sc}'] = np.nan
                metric_row[f'auprc_ratio_{sc}'] = np.nan
                per_score[sc] = (None, None, np.nan)
                continue
            y_score = df_filt[sc].fillna(0).values
            auprc, precision, recall = compute_auprc(y_true, y_score)
            metric_row[f'auprc_{sc}'] = auprc
            metric_row[f'auprc_ratio_{sc}'] = (
                auprc / positive_rate if (not np.isnan(auprc) and positive_rate > 0)
                else np.nan
            )
            per_score[sc] = (precision, recall, auprc)
        rows_metrics.append(metric_row)
        pr_data.append((pert, per_score))
        filtered_dfs.append((pert, df_filt))

        sample_strs = [f'{get_label(sc)}={metric_row[f"auprc_{sc}"]:.3f}'
                       for sc in score_cols if not np.isnan(metric_row[f'auprc_{sc}'])]
        print(f'  {pert}: n_pos={n_pos:5d}, n_neg={n_neg:6d}, '
              f'pos_rate={positive_rate:.4f} | {", ".join(sample_strs)}')

    return (pd.DataFrame(rows_counts),
            pd.DataFrame(rows_metrics),
            pr_data,
            filtered_dfs)



def aggregate_metrics(metrics_df: pd.DataFrame, score_cols: List[str],
                       min_pos: int = 0) -> pd.DataFrame:
    df = metrics_df.copy()
    if min_pos > 0:
        df = df[df['n_pos'] >= min_pos]
    rows = []
    for sc in score_cols:
        col_a = f'auprc_{sc}'
        col_r = f'auprc_ratio_{sc}'
        if col_a not in df.columns:
            continue
        vals_a = df[col_a]
        vals_r = df[col_r]
        pos_rate = df['positive_rate']
        weights = df['n_pos']
        valid = vals_a.notna()
        valid_r = vals_r.notna() & np.isfinite(vals_r)
        valid_pooled = valid & pos_rate.notna() & (pos_rate > 0)

        if valid.sum() > 0 and weights[valid].sum() > 0:
            wm = float(np.average(vals_a[valid], weights=weights[valid]))
        else:
            wm = np.nan
        mm = float(vals_a[valid].mean()) if valid.sum() > 0 else np.nan

        if valid_r.sum() > 0 and weights[valid_r].sum() > 0:
            wm_r = float(np.average(vals_r[valid_r], weights=weights[valid_r]))
        else:
            wm_r = np.nan
        mm_r = float(vals_r[valid_r].mean()) if valid_r.sum() > 0 else np.nan

        if valid_pooled.sum() > 0:
            mm_a_pool = float(vals_a[valid_pooled].mean())
            mm_p_pool = float(pos_rate[valid_pooled].mean())
            mm_ratio_pooled = (mm_a_pool / mm_p_pool) if mm_p_pool > 0 else np.nan
            w_v = weights[valid_pooled]
            if w_v.sum() > 0:
                wm_a_pool = float(np.average(vals_a[valid_pooled], weights=w_v))
                wm_p_pool = float(np.average(pos_rate[valid_pooled], weights=w_v))
                wm_ratio_pooled = (wm_a_pool / wm_p_pool) if wm_p_pool > 0 else np.nan
            else:
                wm_ratio_pooled = np.nan
        else:
            mm_ratio_pooled = np.nan
            wm_ratio_pooled = np.nan

        rows.append({
            'score': sc,
            'label': get_label(sc),
            'n_perts': int(valid.sum()),
            'total_n_pos': int(weights[valid].sum()) if valid.sum() > 0 else 0,
            'weighted_mean_auprc': wm,
            'macro_mean_auprc': mm,
            'weighted_mean_auprc_ratio': wm_r,
            'macro_mean_auprc_ratio': mm_r,
            'weighted_mean_auprc_ratio_pooled': wm_ratio_pooled,
            'macro_mean_auprc_ratio_pooled': mm_ratio_pooled,
        })
    return pd.DataFrame(rows)



DISTANCE_STRATA = [
    ('promoter_0_1kb',    0,       1000,            '0-1kb (promoter)'),
    ('proximal_1_10kb',   1000,    10000,           '1-10kb (proximal)'),
    ('distal_10_100kb',   10000,   100000,          '10-100kb (distal)'),
    ('very_distal_100kb', 100000,  float('inf'),    '>100kb (very distal)'),
]


def compute_per_stratum_per_pert(filtered_dfs, score_cols: List[str],
                                   n_bootstrap: int = 0,
                                   bootstrap_seed: int = 42) -> pd.DataFrame:
    dist_col = 'tss_distance_bp'
    rows = []
    for pert, df in filtered_dfs:
        if dist_col not in df.columns or 'atac_overlap' not in df.columns:
            continue
        df = df.dropna(subset=[dist_col])
        for stratum, lo, hi, label in DISTANCE_STRATA:
            sub = df[(df[dist_col] >= lo) & (df[dist_col] < hi)]
            if len(sub) == 0:
                continue
            y_true = sub['atac_overlap'].astype(int).values
            n_pos = int(y_true.sum())
            n_neg = int(len(y_true) - n_pos)
            if n_pos == 0 or n_neg == 0:
                continue
            pos_rate = n_pos / len(y_true)
            for sc in score_cols:
                if sc not in sub.columns:
                    continue
                y_score = sub[sc].fillna(0).values
                auprc, _, _ = compute_auprc(y_true, y_score)
                ratio = (auprc / pos_rate
                         if (not np.isnan(auprc) and pos_rate > 0) else np.nan)
                row = {
                    'stratum': stratum,
                    'stratum_label': label,
                    'pert': pert,
                    'score': sc,
                    'n_pos': n_pos,
                    'n_neg': n_neg,
                    'positive_rate': pos_rate,
                    'auprc': auprc,
                    'auprc_ratio': ratio,
                }
                if n_bootstrap > 0:
                    ci_lo, ci_hi = bootstrap_auprc_ci(
                        y_true, y_score, n_bootstrap=n_bootstrap,
                        seed=bootstrap_seed)
                    row['auprc_lo'] = ci_lo
                    row['auprc_hi'] = ci_hi
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_stratum(per_stratum_df: pd.DataFrame, score_cols: List[str],
                            min_pos: int = 0) -> pd.DataFrame:
    df = per_stratum_df.copy()
    if min_pos > 0:
        df = df[df['n_pos'] >= min_pos]
    rows = []
    for stratum, lo, hi, label in DISTANCE_STRATA:
        for sc in score_cols:
            sub = df[(df['stratum'] == stratum) & (df['score'] == sc)]
            if sub.empty:
                continue
            for metric in ['auprc', 'auprc_ratio']:
                vals = sub[metric]
                if metric == 'auprc_ratio':
                    valid = vals.notna() & np.isfinite(vals)
                else:
                    valid = vals.notna()
                if valid.sum() == 0:
                    continue
                v = vals[valid]
                w = sub.loc[valid, 'n_pos']
                wm = (float(np.average(v, weights=w))
                      if w.sum() > 0 else np.nan)
                if valid.sum() > 1:
                    sem = float(v.std(ddof=1) / np.sqrt(valid.sum()))
                else:
                    sem = np.nan
                rows.append({
                    'stratum': stratum,
                    'stratum_label': label,
                    'score': sc,
                    'metric': metric,
                    'n_perts': int(valid.sum()),
                    'total_n_pos': int(w.sum()),
                    'weighted_mean': wm,
                    'macro_mean': float(v.mean()),
                    'macro_sem': sem,
                    'median': float(v.median()),
                })

            valid_pool = (sub['auprc'].notna() & sub['positive_rate'].notna()
                          & (sub['positive_rate'] > 0))
            if valid_pool.sum() == 0:
                continue
            a = sub.loc[valid_pool, 'auprc']
            p = sub.loc[valid_pool, 'positive_rate']
            w = sub.loc[valid_pool, 'n_pos']
            macro_pool = (float(a.mean()) / float(p.mean())
                          if float(p.mean()) > 0 else np.nan)
            if w.sum() > 0:
                wm_a = float(np.average(a, weights=w))
                wm_p = float(np.average(p, weights=w))
                weighted_pool = wm_a / wm_p if wm_p > 0 else np.nan
            else:
                weighted_pool = np.nan
            rows.append({
                'stratum': stratum,
                'stratum_label': label,
                'score': sc,
                'metric': 'auprc_ratio_pooled',
                'n_perts': int(valid_pool.sum()),
                'total_n_pos': int(w.sum()),
                'weighted_mean': weighted_pool,
                'macro_mean': macro_pool,
                'macro_sem': np.nan,
                'median': np.nan,
            })
    return pd.DataFrame(rows)



def compute_wilcoxon_vs_attr(per_strat_df: pd.DataFrame, score_cols: List[str],
                              reference_score: str = 'attr_top10_score',
                              min_pos: int = 0) -> pd.DataFrame:
    df = per_strat_df.copy()
    if min_pos > 0:
        df = df[df['n_pos'] >= min_pos]
    rows = []
    for stratum, lo, hi, label in DISTANCE_STRATA:
        sub_strat = df[df['stratum'] == stratum]
        if sub_strat.empty:
            continue
        ref = sub_strat[sub_strat['score'] == reference_score][['pert', 'auprc']]
        if ref.empty:
            continue
        ref = ref.rename(columns={'auprc': 'auprc_ref'})
        for sc in score_cols:
            if sc == reference_score:
                continue
            other = sub_strat[sub_strat['score'] == sc][['pert', 'auprc']]
            if other.empty:
                continue
            merged = ref.merge(other, on='pert', how='inner').dropna(
                subset=['auprc_ref', 'auprc'])
            if len(merged) < 3:
                rows.append({
                    'stratum': stratum,
                    'stratum_label': label,
                    'reference': reference_score,
                    'score': sc,
                    'n_pairs': len(merged),
                    'mean_diff': (float((merged['auprc_ref'] - merged['auprc']).mean())
                                  if len(merged) else np.nan),
                    'wilcoxon_p_greater': np.nan,
                })
                continue
            diff = (merged['auprc_ref'] - merged['auprc']).values
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    p_val = float(wilcoxon(merged['auprc_ref'].values,
                                            merged['auprc'].values,
                                            alternative='greater').pvalue)
                except ValueError:
                    p_val = np.nan
            rows.append({
                'stratum': stratum,
                'stratum_label': label,
                'reference': reference_score,
                'score': sc,
                'n_pairs': len(merged),
                'mean_diff': float(diff.mean()),
                'wilcoxon_p_greater': p_val,
            })
    return pd.DataFrame(rows)


def _stars(p):
    if not np.isfinite(p):
        return ''
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'



def _nice_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.1
    padded = value * 1.1
    if padded < 0.1:
        step = 0.01
    elif padded < 1.0:
        step = 0.05
    elif padded < 10.0:
        step = 0.5
    else:
        step = 1.0
    return float(np.ceil(padded / step) * step)


def plot_pr_curves(pr_data, output_dir: str, score_cols: List[str]):
    safe_makedirs(output_dir)
    global_max_p = 0.0
    for _, per_score in pr_data:
        for sc in score_cols:
            triple = per_score.get(sc)
            if triple is None or triple[0] is None:
                continue
            global_max_p = max(global_max_p, float(np.nanmax(triple[0])))
    ymax = min(1.0, _nice_upper(global_max_p)) if global_max_p > 0 else 1.0

    for pert, per_score in pr_data:
        out = os.path.join(output_dir, f'pr_curve_{pert}.svg')
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        for sc in score_cols:
            triple = per_score.get(sc)
            if triple is None or triple[0] is None:
                continue
            precision, recall, auprc = triple
            if precision is None or np.isnan(auprc):
                continue
            ax.plot(recall, precision, color=get_color(sc), lw=2,
                    label=f'{get_label(sc)} ({auprc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR curve (DEG-filtered): {pert}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, ymax])
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def plot_per_pert_bar(metrics_df: pd.DataFrame, score_cols: List[str],
                       output_path: str, min_pos: int):
    df = metrics_df.copy()
    if min_pos > 0:
        df = df[df['n_pos'] >= min_pos]
    if df.empty:
        print(f'[WARN] no rows for per-pert bar plot at min_pos={min_pos}')
        return
    sort_col = 'auprc_attr_top10_score'
    if sort_col not in df.columns:
        sort_col = f'auprc_{score_cols[0]}'
    df = df.sort_values(sort_col, ascending=False)

    n = len(df)
    fig_w = max(14, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=150)
    x = np.arange(n)
    n_sc = len(score_cols)
    width = 0.8 / n_sc
    max_value = 0.0
    for i, sc in enumerate(score_cols):
        col = f'auprc_{sc}'
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=get_label(sc),
               color=get_color(sc), alpha=0.85)
        if vals.size:
            max_value = max(max_value, float(np.nanmax(vals)))
    ax.set_xticks(x + width * (n_sc / 2))
    labels = [f'{r["pert"]}\n(P={r["n_pos"]:.0f})' for _, r in df.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_xlabel('Perturbation (n_pos)')
    ax.set_ylabel('AUPRC')
    title_suffix = '' if min_pos == 0 else f', n_pos>={min_pos}'
    ax.set_title(f'Per-pert AUPRC (DEG-filtered{title_suffix})')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim([0, _nice_upper(max_value)])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_aggregate(agg_df: pd.DataFrame, output_path: str, scope_label: str):
    if agg_df.empty:
        print(f'[WARN] aggregate plot skipped ({scope_label}): empty')
        return
    fig, axes = plt.subplots(1, 3, figsize=(28, 6), dpi=150)
    score_order = list(agg_df['score'])
    colors = [get_color(s) for s in score_order]
    labels = [get_label(s) for s in score_order]
    x = np.arange(len(score_order))
    width = 0.4

    ax = axes[0]
    ax.bar(x - width / 2, agg_df['weighted_mean_auprc'].fillna(0).values, width,
           label='Weighted mean (by n_pos)', color=colors, edgecolor='black', linewidth=1.0)
    ax.bar(x + width / 2, agg_df['macro_mean_auprc'].fillna(0).values, width,
           label='Macro mean (equal pert)', color=colors, edgecolor='black',
           linewidth=1.0, hatch='///', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('AUPRC')
    ax.set_title(f'AUPRC (DEG-filtered, {scope_label})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width / 2, agg_df['weighted_mean_auprc_ratio'].fillna(0).values, width,
           label='Weighted mean (by n_pos)', color=colors, edgecolor='black', linewidth=1.0)
    ax.bar(x + width / 2, agg_df['macro_mean_auprc_ratio'].fillna(0).values, width,
           label='Macro mean (equal pert)', color=colors, edgecolor='black',
           linewidth=1.0, hatch='///', alpha=0.6)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='random (=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('mean(AUPRC / positive_rate)')
    ax.set_title(f'AUPRC ratio — per-pert (DEG-filtered, {scope_label})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x - width / 2, agg_df['weighted_mean_auprc_ratio_pooled'].fillna(0).values, width,
           label='Weighted mean (by n_pos)', color=colors, edgecolor='black', linewidth=1.0)
    ax.bar(x + width / 2, agg_df['macro_mean_auprc_ratio_pooled'].fillna(0).values, width,
           label='Macro mean (equal pert)', color=colors, edgecolor='black',
           linewidth=1.0, hatch='///', alpha=0.6)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='random (=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('mean(AUPRC) / mean(positive_rate)')
    ax.set_title(f'AUPRC ratio — pooled (DEG-filtered, {scope_label})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_distance_stratified_single_agg(
        per_strat_df: pd.DataFrame, agg_strat_df: pd.DataFrame,
        score_cols: List[str], metric: str, agg_method: str,
        output_path: str, scope_label: str, min_pos: int = 0,
        wilcoxon_df: pd.DataFrame = None,
        reference_score: str = 'attr_top10_score'):
    if agg_strat_df.empty:
        print(f'[WARN] single-agg plot skipped ({metric}/{agg_method}, {scope_label}): empty')
        return
    sub = agg_strat_df[agg_strat_df['metric'] == metric]
    if sub.empty:
        return

    show_errors = (agg_method == 'macro_mean' and metric == 'auprc'
                   and 'macro_sem' in sub.columns)
    show_stars = (show_errors and wilcoxon_df is not None
                  and not wilcoxon_df.empty)

    bar_vals = sub[agg_method].dropna()
    if metric == 'auprc_ratio':
        bar_vals = bar_vals[np.isfinite(bar_vals)]
    ymax = _nice_upper(float(bar_vals.max())) if len(bar_vals) else 1.0
    if metric == 'auprc_ratio':
        ymax = max(ymax, 1.05)
    if show_errors:
        ymax = ymax * 1.12

    n_strata = len(DISTANCE_STRATA)
    fig, axes = plt.subplots(1, n_strata, figsize=(6 * n_strata, 6.5),
                              dpi=150, sharey=True)
    if n_strata == 1:
        axes = [axes]
    x = np.arange(len(score_cols))
    width = 0.6

    for ax, (stratum, lo, hi, label) in zip(axes, DISTANCE_STRATA):
        s = sub[sub['stratum'] == stratum]
        if s.empty:
            ax.set_title(f'{label}\nNo data', fontsize=18)
            ax.set_ylim([0, ymax])
            continue
        heights, errs, colors_b = [], [], []
        for sc in score_cols:
            row = s[s['score'] == sc]
            if row.empty or pd.isna(row[agg_method].iloc[0]):
                heights.append(0.0)
                errs.append(0.0)
            else:
                heights.append(float(row[agg_method].iloc[0]))
                if show_errors and 'macro_sem' in row.columns:
                    sem = row['macro_sem'].iloc[0]
                    errs.append(float(sem) if pd.notna(sem) else 0.0)
                else:
                    errs.append(0.0)
            colors_b.append(get_color(sc))
        if show_errors:
            ax.bar(x, heights, width, color=colors_b, alpha=0.85,
                   edgecolor='black', linewidth=0.7,
                   yerr=errs, capsize=4,
                   error_kw={'elinewidth': 1.0, 'ecolor': 'black'})
        else:
            ax.bar(x, heights, width, color=colors_b, alpha=0.85,
                   edgecolor='black', linewidth=0.7)

        if show_stars:
            wsub = wilcoxon_df[wilcoxon_df['stratum'] == stratum]
            for xi, sc in enumerate(score_cols):
                if sc == reference_score:
                    continue
                wrow = wsub[wsub['score'] == sc]
                if wrow.empty:
                    continue
                p = wrow['wilcoxon_p_greater'].iloc[0]
                star = _stars(p)
                if not star:
                    continue
                bar_top = heights[xi] + (errs[xi] if errs[xi] else 0)
                ax.text(xi, bar_top + ymax * 0.02, star,
                        ha='center', va='bottom', fontsize=14,
                        color='black')

        if metric == 'auprc_ratio':
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(sc) for sc in score_cols],
                           rotation=35, ha='right', fontsize=15)
        ax.set_title(label, fontsize=18)
        ax.set_ylim([0, ymax])
        ax.grid(True, alpha=0.3, axis='y')

    if metric == 'auprc':
        base_ylabel = 'AUPRC'
    elif metric == 'auprc_ratio':
        base_ylabel = 'AUPRC / positive_rate'
    else:  # auprc_ratio_pooled
        base_ylabel = 'mean(AUPRC) / mean(positive_rate)'
    agg_word = 'mean' if agg_method == 'macro_mean' else 'weighted mean'
    axes[0].set_ylabel(f'{base_ylabel}\n{agg_word}\nacross perturbations',
                       fontsize=20)
    fig.suptitle(f'{base_ylabel} by TSS-distance stratum (DEG-filtered, {scope_label})',
                 fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path)
    plt.close()


def plot_distance_stratified_mean_vs_weighted(
        per_strat_df: pd.DataFrame, agg_strat_df: pd.DataFrame,
        score_cols: List[str], metric: str,
        output_path: str, scope_label: str, min_pos: int = 0):
    if agg_strat_df.empty:
        print(f'[WARN] mean-vs-weighted plot skipped ({metric}, {scope_label}): empty')
        return
    sub = agg_strat_df[agg_strat_df['metric'] == metric]
    if sub.empty:
        return

    bars = pd.concat([sub['macro_mean'].dropna(), sub['weighted_mean'].dropna()])
    if metric == 'auprc_ratio':
        bars = bars[np.isfinite(bars)]
    ymax = _nice_upper(float(bars.max())) if len(bars) else 1.0
    if metric == 'auprc_ratio':
        ymax = max(ymax, 1.05)

    n_strata = len(DISTANCE_STRATA)
    fig, axes = plt.subplots(1, n_strata, figsize=(6.5 * n_strata, 6.5),
                              dpi=150, sharey=True)
    if n_strata == 1:
        axes = [axes]
    x = np.arange(len(score_cols))
    width = 0.38
    aggs = [('macro_mean',    'Mean',                   -width / 2, 0.95),
            ('weighted_mean', 'Weighted mean (n_pos)',   width / 2, 0.55)]

    for ax, (stratum, lo, hi, label) in zip(axes, DISTANCE_STRATA):
        s = sub[sub['stratum'] == stratum]
        if s.empty:
            ax.set_title(f'{label}\nNo data', fontsize=18)
            ax.set_ylim([0, ymax])
            continue
        for col, agg_label, offset, alpha in aggs:
            heights, colors_b = [], []
            for sc in score_cols:
                row = s[s['score'] == sc]
                if row.empty or pd.isna(row[col].iloc[0]):
                    heights.append(0.0)
                else:
                    heights.append(float(row[col].iloc[0]))
                colors_b.append(get_color(sc))
            ax.bar(x + offset, heights, width, color=colors_b,
                   alpha=alpha, edgecolor='black', linewidth=0.6)
        if metric == 'auprc_ratio':
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(sc) for sc in score_cols],
                           rotation=35, ha='right', fontsize=15)
        ax.set_title(label, fontsize=18)
        ax.set_ylim([0, ymax])
        ax.grid(True, alpha=0.3, axis='y')

    if metric == 'auprc':
        base_ylabel = 'AUPRC'
    elif metric == 'auprc_ratio':
        base_ylabel = 'AUPRC / positive_rate'
    else:  # auprc_ratio_pooled
        base_ylabel = 'mean(AUPRC) / mean(positive_rate)'
    axes[0].set_ylabel(f'{base_ylabel}\nmean\nacross perturbations', fontsize=20)
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.95, label='Mean'),
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.55,
                      label='Weighted mean (n_pos)'),
    ]
    axes[-1].legend(handles=legend_elems, loc='upper right', fontsize=13)
    fig.suptitle(f'{base_ylabel} by TSS-distance stratum (DEG-filtered, {scope_label})',
                 fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path)
    plt.close()


def plot_per_pert_bar_by_distance(per_strat_df: pd.DataFrame,
                                    score_cols: List[str], output_path: str,
                                    scope_label: str, min_pos: int = 0):
    if per_strat_df.empty:
        print(f'[WARN] per-pert distance plot skipped ({scope_label}): empty')
        return
    df = per_strat_df.copy()
    if min_pos > 0:
        df = df[df['n_pos'] >= min_pos]
    if df.empty:
        print(f'[WARN] per-pert distance plot skipped ({scope_label}): no rows after min_pos filter')
        return

    sort_score = ('attr_top10_score' if 'attr_top10_score' in df['score'].unique()
                  else df['score'].iloc[0])
    sort_df = df[df['score'] == sort_score]
    if sort_df.empty:
        global_order = list(df['pert'].drop_duplicates())
    else:
        order_keys = sort_df.groupby('pert')['auprc'].mean().sort_values(
            ascending=False)
        rest = [p for p in df['pert'].drop_duplicates() if p not in order_keys.index]
        global_order = list(order_keys.index) + rest

    total_npos = df.groupby('pert')['n_pos'].sum().reindex(global_order).fillna(0)

    vals_all = df['auprc'].dropna()
    ymax = _nice_upper(float(vals_all.max())) if len(vals_all) else 1.0

    n_strata = len(DISTANCE_STRATA)
    n_perts = len(global_order)
    fig_w = max(14, n_perts * 0.75)
    fig, axes = plt.subplots(n_strata, 1, figsize=(fig_w, 5.5 * n_strata),
                              dpi=150, sharey=True, sharex=True)
    if n_strata == 1:
        axes = [axes]

    n_sc = len(score_cols)
    width = 0.8 / n_sc
    x = np.arange(n_perts)

    has_ci = ('auprc_lo' in df.columns) and ('auprc_hi' in df.columns)

    for ax, (stratum, lo, hi, label) in zip(axes, DISTANCE_STRATA):
        sub = df[df['stratum'] == stratum]
        if sub.empty:
            ax.set_title(f'{label}: no data', fontsize=18)
            ax.set_ylim([0, ymax])
            continue
        piv = sub.pivot_table(index='pert', columns='score', values='auprc',
                              aggfunc='first').reindex(global_order)
        if has_ci:
            piv_lo = sub.pivot_table(index='pert', columns='score',
                                      values='auprc_lo', aggfunc='first'
                                      ).reindex(global_order)
            piv_hi = sub.pivot_table(index='pert', columns='score',
                                      values='auprc_hi', aggfunc='first'
                                      ).reindex(global_order)
        for i, sc in enumerate(score_cols):
            if sc not in piv.columns:
                continue
            vals = piv[sc].fillna(0).values
            if has_ci and sc in piv_lo.columns and sc in piv_hi.columns:
                lo_vals = piv_lo[sc].values
                hi_vals = piv_hi[sc].values
                lower = np.where(np.isnan(lo_vals), 0, vals - np.nan_to_num(lo_vals, nan=vals))
                upper = np.where(np.isnan(hi_vals), 0, np.nan_to_num(hi_vals, nan=vals) - vals)
                lower = np.clip(lower, 0, None)
                upper = np.clip(upper, 0, None)
                ax.bar(x + i * width, vals, width, label=get_label(sc),
                       color=get_color(sc), alpha=0.85,
                       yerr=[lower, upper], capsize=2,
                       error_kw={'elinewidth': 0.6, 'ecolor': 'black'})
            else:
                ax.bar(x + i * width, vals, width, label=get_label(sc),
                       color=get_color(sc), alpha=0.85)
        ax.set_ylabel('AUPRC')
        ax.set_title(label, fontsize=18)
        ax.set_ylim([0, ymax])
        ax.grid(True, alpha=0.3, axis='y')
        if stratum == DISTANCE_STRATA[0][0]:
            ax.legend(loc='upper right', ncol=2, fontsize=14)

    xticklabels = [f'{p}\n(P_total={int(total_npos[p])})' for p in global_order]
    axes[-1].set_xticks(x + width * (n_sc / 2))
    axes[-1].set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=13)
    axes[-1].set_xlabel('Perturbation (total n_pos across strata)')
    fig.suptitle(f'Per-pert AUPRC by TSS-distance stratum (DEG-filtered, {scope_label})',
                 fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path)
    plt.close()


def plot_distance_stratified(agg_strat_df: pd.DataFrame, score_cols: List[str],
                               metric: str, output_path: str, scope_label: str):
    if agg_strat_df.empty:
        print(f'[WARN] distance-stratified plot skipped: empty ({metric}, {scope_label})')
        return
    sub = agg_strat_df[agg_strat_df['metric'] == metric]
    if sub.empty:
        return

    candidates = pd.concat([sub['weighted_mean'], sub['macro_mean'], sub['median']])
    candidates = candidates[np.isfinite(candidates)]
    ymax = _nice_upper(float(candidates.max())) if len(candidates) else 1.0
    if metric == 'auprc_ratio':
        ymax = max(ymax, 1.05)  # ensure baseline=1 is visible

    n_strata = len(DISTANCE_STRATA)
    fig, axes = plt.subplots(1, n_strata, figsize=(6 * n_strata, 6),
                              dpi=150, sharey=True)
    if n_strata == 1:
        axes = [axes]

    n_sc = len(score_cols)
    x = np.arange(n_sc)
    width = 0.27
    aggs = [('weighted_mean', 'Weighted (n_pos)', 0.95),
            ('macro_mean',    'Macro mean',       0.7),
            ('median',        'Median',           0.45)]

    for ax, (stratum, lo, hi, label) in zip(axes, DISTANCE_STRATA):
        s = sub[sub['stratum'] == stratum]
        if s.empty:
            ax.set_title(f'{label}\nNo data', fontsize=18)
            ax.set_ylim([0, ymax])
            continue
        for offset, (col, agg_label, alpha) in zip([-width, 0, width], aggs):
            heights = []
            colors_b = []
            for sc in score_cols:
                row = s[s['score'] == sc]
                if row.empty or pd.isna(row[col].iloc[0]):
                    heights.append(0.0)
                else:
                    heights.append(float(row[col].iloc[0]))
                colors_b.append(get_color(sc))
            ax.bar(x + offset, heights, width, color=colors_b,
                   alpha=alpha, edgecolor='black', linewidth=0.6,
                   label=agg_label if stratum == DISTANCE_STRATA[0][0] else None)
        if metric == 'auprc_ratio':
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(sc) for sc in score_cols],
                           rotation=35, ha='right', fontsize=15)
        ax.set_title(label, fontsize=18)
        ax.set_ylim([0, ymax])
        ax.grid(True, alpha=0.3, axis='y')

    ylabel = 'AUPRC' if metric == 'auprc' else 'AUPRC / positive_rate'
    axes[0].set_ylabel(ylabel, fontsize=20)

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.95, label='Weighted (n_pos)'),
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7,  label='Macro mean'),
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.45, label='Median'),
    ]
    axes[-1].legend(handles=legend_elems, loc='upper right', fontsize=14)

    fig.suptitle(f'{ylabel} by TSS-distance stratum (DEG-filtered, {scope_label})',
                 fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path)
    plt.close()
