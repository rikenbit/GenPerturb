#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Dict, List

import numpy as np
import pandas as pd



_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DEG_FILTERED_PATH = os.path.join(_HERE, '_auprc_eval_helpers.py')
_spec = importlib.util.spec_from_file_location('eval_deg_filtered', _EVAL_DEG_FILTERED_PATH)
eval_deg_filtered = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_deg_filtered)



def load_martin_degs_fc(xlsx_path: str, fc_min: float) -> Dict[str, set]:
    df = pd.read_excel(xlsx_path, sheet_name='TF_sensitive_genes', header=2)
    df = df.dropna(subset=['gene_ID', 'perturbation_name', 'log2FC'])
    if fc_min > 0:
        df = df[df['log2FC'].abs() >= fc_min]
    return {tf: set(g['gene_ID'].astype(str).unique())
            for tf, g in df.groupby('perturbation_name')}


def load_metzner_degs_fc(xlsx_path: str, q_max: float,
                          zscore_min: float) -> Dict[str, set]:
    df = pd.read_excel(xlsx_path, sheet_name='genes')
    df['q'] = pd.to_numeric(df['q'], errors='coerce')
    df['zscore'] = pd.to_numeric(df['zscore'], errors='coerce')
    df = df.dropna(subset=['gene', 'guide_target', 'q', 'zscore'])
    df = df[df['q'] <= q_max]
    if zscore_min > 0:
        df = df[df['zscore'].abs() >= zscore_min]
    return {tf: set(g['gene'].astype(str).unique())
            for tf, g in df.groupby('guide_target')}



def compute_metzner_zscore_thresholds(martin_xlsx: str, metzner_xlsx: str,
                                       q_max: float,
                                       martin_fc_thresholds: List[float]) -> Dict[float, float]:
    m = pd.read_excel(martin_xlsx, sheet_name='TF_sensitive_genes', header=2)
    m_fc = m['log2FC'].dropna().abs().values
    z = pd.read_excel(metzner_xlsx, sheet_name='genes')
    z['q'] = pd.to_numeric(z['q'], errors='coerce')
    z['zscore'] = pd.to_numeric(z['zscore'], errors='coerce')
    z = z.dropna(subset=['q', 'zscore'])
    z = z[z['q'] <= q_max]
    z_abs = z['zscore'].abs().values
    out: Dict[float, float] = {}
    for t in martin_fc_thresholds:
        if t <= 0:
            out[t] = 0.0
            continue
        keep_frac = float((m_fc >= t).mean())
        q_target = max(0.0, min(1.0, 1.0 - keep_frac))
        out[t] = float(np.quantile(z_abs, q_target))
    return out



def run_pipeline(scored_dir: str, out_dir: str, degs_by_pert: Dict[str, set],
                  min_pos: int, threshold_tag: str,
                  n_bootstrap: int = 1000, bootstrap_seed: int = 42):
    eval_deg_filtered.safe_makedirs(out_dir)
    pr_dir = os.path.join(out_dir, 'pr_curves')
    eval_deg_filtered.safe_makedirs(pr_dir)

    print(f'\n[INFO] === FC threshold {threshold_tag} === out={out_dir}')
    counts_df, metrics_df, pr_data, filtered_dfs = eval_deg_filtered.evaluate_scored_peaks(
        scored_dir, degs_by_pert, eval_deg_filtered.SCORE_COLUMNS)

    counts_df.to_csv(os.path.join(out_dir, 'pos_neg_counts_per_pert.tsv'),
                      sep='\t', index=False)
    if metrics_df.empty:
        print('[WARN] No usable filtered set for this threshold — skipping plots.')
        return metrics_df
    metrics_df.to_csv(os.path.join(out_dir, 'auprc_summary_per_pert.tsv'),
                       sep='\t', index=False)

    for scope_name, mp in [('all', 0), ('min_pos10', min_pos)]:
        bar_path = os.path.join(out_dir, f'auprc_per_pert_{scope_name}.svg')
        eval_deg_filtered.plot_per_pert_bar(metrics_df, eval_deg_filtered.SCORE_COLUMNS,
                                   bar_path, mp)
        agg_df = eval_deg_filtered.aggregate_metrics(metrics_df, eval_deg_filtered.SCORE_COLUMNS,
                                            min_pos=mp)
        agg_df.to_csv(os.path.join(out_dir, f'auprc_aggregated_{scope_name}.tsv'),
                       sep='\t', index=False)
        eval_deg_filtered.plot_aggregate(agg_df,
                                os.path.join(out_dir, f'auprc_aggregate_{scope_name}.svg'),
                                scope_label=scope_name)

    eval_deg_filtered.plot_pr_curves(pr_data, pr_dir, eval_deg_filtered.SCORE_COLUMNS)

    if n_bootstrap > 0:
        print(f'[INFO] Bootstrapping per-pert AUPRC 95% CI '
              f'(n={n_bootstrap}, seed={bootstrap_seed})...')
    per_strat_df = eval_deg_filtered.compute_per_stratum_per_pert(
        filtered_dfs, eval_deg_filtered.SCORE_COLUMNS,
        n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
    per_strat_df.to_csv(os.path.join(out_dir, 'auprc_per_pert_by_distance.tsv'),
                         sep='\t', index=False)

    for scope_name, mp in [('all', 0), ('min_pos10', min_pos)]:
        wdf = eval_deg_filtered.compute_wilcoxon_vs_attr(
            per_strat_df, eval_deg_filtered.SCORE_COLUMNS,
            reference_score='attr_top10_score', min_pos=mp)
        wdf.to_csv(os.path.join(
            out_dir, f'wilcoxon_attr_vs_baselines_{scope_name}.tsv'),
            sep='\t', index=False)

    for scope_name, mp in [('all', 0), ('min_pos10', min_pos)]:
        agg_strat = eval_deg_filtered.aggregate_per_stratum(per_strat_df,
                                                    eval_deg_filtered.SCORE_COLUMNS,
                                                    min_pos=mp)
        agg_strat.to_csv(os.path.join(
            out_dir, f'auprc_aggregated_by_distance_{scope_name}.tsv'),
            sep='\t', index=False)

        for metric in ['auprc', 'auprc_ratio', 'auprc_ratio_pooled']:
            if metric != 'auprc_ratio_pooled':
                eval_deg_filtered.plot_distance_stratified(
                    agg_strat, eval_deg_filtered.SCORE_COLUMNS, metric,
                    os.path.join(out_dir, f'{metric}_by_distance_{scope_name}.svg'),
                    scope_label=scope_name)
            eval_deg_filtered.plot_distance_stratified_single_agg(
                per_strat_df, agg_strat, eval_deg_filtered.SCORE_COLUMNS, metric,
                agg_method='macro_mean',
                output_path=os.path.join(
                    out_dir, f'{metric}_by_distance_{scope_name}_mean.svg'),
                scope_label=scope_name, min_pos=mp,
                wilcoxon_df=None)  # SEM whiskers only; no significance stars
            eval_deg_filtered.plot_distance_stratified_single_agg(
                per_strat_df, agg_strat, eval_deg_filtered.SCORE_COLUMNS, metric,
                agg_method='weighted_mean',
                output_path=os.path.join(
                    out_dir, f'{metric}_by_distance_{scope_name}_weighted_mean.svg'),
                scope_label=scope_name, min_pos=mp)
            eval_deg_filtered.plot_distance_stratified_mean_vs_weighted(
                per_strat_df, agg_strat, eval_deg_filtered.SCORE_COLUMNS, metric,
                output_path=os.path.join(
                    out_dir,
                    f'{metric}_by_distance_{scope_name}_mean_vs_weighted.svg'),
                scope_label=scope_name, min_pos=mp)

        eval_deg_filtered.plot_per_pert_bar_by_distance(
            per_strat_df, eval_deg_filtered.SCORE_COLUMNS,
            os.path.join(out_dir, f'auprc_per_pert_by_distance_{scope_name}.svg'),
            scope_label=scope_name, min_pos=mp)

    return metrics_df



import matplotlib.pyplot as plt  # noqa: E402  (matplotlib already configured by the helper module)


def plot_threshold_sweep(per_threshold_metrics: Dict[str, pd.DataFrame],
                           score_cols: List[str], output_dir: str,
                           min_pos: int):
    if not per_threshold_metrics:
        return
    eval_deg_filtered.safe_makedirs(output_dir)

    rows = []
    for tag, m in per_threshold_metrics.items():
        if m.empty:
            continue
        agg = eval_deg_filtered.aggregate_metrics(m, score_cols, min_pos=min_pos)
        for _, r in agg.iterrows():
            rows.append({
                'fc_tag': tag,
                'score': r['score'],
                'label': r['label'],
                'n_perts': r['n_perts'],
                'total_n_pos': r['total_n_pos'],
                'macro_mean_auprc': r['macro_mean_auprc'],
                'weighted_mean_auprc': r['weighted_mean_auprc'],
                'macro_mean_auprc_ratio_pooled': r['macro_mean_auprc_ratio_pooled'],
                'weighted_mean_auprc_ratio_pooled': r['weighted_mean_auprc_ratio_pooled'],
            })
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, 'fc_sweep_summary.tsv'),
                    sep='\t', index=False)

    metrics_to_plot = [
        ('macro_mean_auprc',                 'AUPRC\nmean across perturbations'),
        ('weighted_mean_auprc',              'AUPRC\nweighted mean (n_pos)\nacross perturbations'),
        ('macro_mean_auprc_ratio_pooled',    'mean(AUPRC)/mean(positive_rate)\nmean across perturbations'),
        ('weighted_mean_auprc_ratio_pooled', 'mean(AUPRC)/mean(positive_rate)\nweighted mean (n_pos)\nacross perturbations'),
    ]
    fc_tags = list(per_threshold_metrics.keys())  # preserves insertion order
    x = np.arange(len(fc_tags))

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6.5 * n_metrics, 6.5), dpi=150)
    if n_metrics == 1:
        axes = [axes]
    for ax, (col, ylabel) in zip(axes, metrics_to_plot):
        for sc in score_cols:
            sub = summary[summary['score'] == sc].set_index('fc_tag').reindex(fc_tags)
            ax.plot(x, sub[col].values, marker='o', lw=2,
                    color=eval_deg_filtered.get_color(sc), label=eval_deg_filtered.get_label(sc))
        ax.set_xticks(x)
        ax.set_xticklabels(fc_tags, rotation=0)
        ax.set_xlabel('FC threshold')
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=14)
    fig.suptitle(f'AUPRC vs FC threshold (DEG-filtered, min_pos>={min_pos})',
                 fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_svg = os.path.join(output_dir, f'fc_sweep_min_pos{min_pos}.svg')
    plt.savefig(out_svg)
    plt.close()
    print(f'[INFO] Saved FC-sweep summary plot: {out_svg}')



def fc_tag(value: float) -> str:
    return f'{value:.2f}'.replace('.', 'p')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=['metzner', 'martin'])
    ap.add_argument('--study_name', required=True)
    ap.add_argument('--study_suffix',
                    default='alphagenome_transfer_epoch100_batch256_adamw5e3')
    ap.add_argument('--cre_root', default='cre_gtgenes')
    ap.add_argument('--figures_root', default='figures')
    ap.add_argument('--metzner_xlsx',
                    default='data/PRJNA1128171/supp/NIHMS2040923-supplement-MMC4.xlsx')
    ap.add_argument('--metzner_q_max', type=float, default=0.1)
    ap.add_argument('--martin_xlsx',
                    default='data/science.ads7951_tables_s1_to_s6/'
                            'science.ads7951_table_s3.xlsx')
    ap.add_argument('--fc_thresholds', default='0,0.1,0.2,0.3',
                    help='Comma-separated Martin |log2FC| thresholds. '
                         'Metzner |zscore| thresholds are auto-derived to '
                         'match per-DEG retention rate.')
    ap.add_argument('--min_pos', type=int, default=10)
    ap.add_argument('--n_bootstrap', type=int, default=1000,
                    help='Within-pert bootstrap iterations for per-pert AUPRC '
                         '95%% CI (Fig S8 whiskers). Set 0 to disable.')
    ap.add_argument('--bootstrap_seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    study = f'{args.study_name}__{args.study_suffix}'
    scored_dir = os.path.join(args.cre_root, study, 'scored_peaks')

    fc_list = [float(t) for t in args.fc_thresholds.split(',')]

    z_map = compute_metzner_zscore_thresholds(
        args.martin_xlsx, args.metzner_xlsx, args.metzner_q_max, fc_list)
    print('[INFO] FC threshold mapping (martin |log2FC| -> metzner |zscore|):')
    for t in fc_list:
        print(f'    {t:.2f} -> {z_map[t]:.4f}')

    per_threshold_metrics: Dict[str, pd.DataFrame] = {}
    for t in fc_list:
        if args.study == 'martin':
            print(f'\n[INFO] Loading Martin DEGs from {args.martin_xlsx} '
                  f'(|log2FC| >= {t:.2f})')
            degs = load_martin_degs_fc(args.martin_xlsx, fc_min=t)
        else:
            zt = z_map[t]
            print(f'\n[INFO] Loading Metzner DEGs from {args.metzner_xlsx} '
                  f'(q <= {args.metzner_q_max}, |zscore| >= {zt:.4f}; '
                  f'martin |log2FC| equivalent = {t:.2f})')
            degs = load_metzner_degs_fc(args.metzner_xlsx,
                                         q_max=args.metzner_q_max,
                                         zscore_min=zt)

        deg_total = sum(len(s) for s in degs.values())
        print(f'  -> {len(degs)} perts, {deg_total} total DEG entries')

        tag = fc_tag(t)
        out_dir = os.path.join(args.figures_root, study,
                                f'auprc_evaluation_deg_fc_filtered_fc{tag}')
        metrics_df = run_pipeline(scored_dir, out_dir, degs,
                                    min_pos=args.min_pos, threshold_tag=tag,
                                    n_bootstrap=args.n_bootstrap,
                                    bootstrap_seed=args.bootstrap_seed)
        per_threshold_metrics[tag] = metrics_df

    sweep_dir = os.path.join(args.figures_root, study,
                              'auprc_evaluation_deg_fc_filtered_sweep')
    plot_threshold_sweep(per_threshold_metrics, eval_deg_filtered.SCORE_COLUMNS,
                          sweep_dir, min_pos=args.min_pos)


if __name__ == '__main__':
    main()
