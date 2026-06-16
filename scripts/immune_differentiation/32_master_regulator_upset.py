from __future__ import annotations

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'svg.fonttype': 'none',
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = 'NormanWeissman2019_filtered_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3'

BASE_DIR = PROJECT_ROOT / 'figures' / RUN_TAG / 'master_regulator'
SCENIC_DIR = BASE_DIR / 'scenic'
OUTPUT_DIR = BASE_DIR
FIGURE_DIR = BASE_DIR
BASE_DIR.mkdir(parents=True, exist_ok=True)
SCENIC_DIR.mkdir(parents=True, exist_ok=True)

STUDY_NAME = RUN_TAG.split('__')[0]
TFMODISCO_DIR = PROJECT_ROOT / 'attribution_analysis' / 'tfmodisco' / RUN_TAG
TFMODISCO_MATRIX_POS = TFMODISCO_DIR / 'tfmodisco_motif_matrix_pos.tsv'
TFMODISCO_MATRIX_NEG = TFMODISCO_DIR / 'tfmodisco_motif_matrix_neg.tsv'
GIMME_BASE = PROJECT_ROOT / 'attribution_analysis' / 'gimme_results' / RUN_TAG
SIGNATURE_FILE = (PROJECT_ROOT / 'figures' / RUN_TAG / 'gene_signature' /
                  'signature_scores.txt')
EXPRESSION_FILE = PROJECT_ROOT / 'data' / f'{STUDY_NAME}.tsv'
AUCELL_FILE = SCENIC_DIR / 'aucell_mean_per_perturbation.tsv'
JASPAR_CLUSTERS = PROJECT_ROOT / 'reference' / 'jaspar' / 'clusters.tab'

GIMME_BED_TYPES = {
    'gimme_attribution': 'attribution',
    'gimme_re2g':        're2g',
    'gimme_abc':         'abc_score',
    'gimme_tss_1kbp':    'tss_1kbp',
}
EXPRESSION_MIN_MEAN = 0.5
MIN_NONZERO_FRAC = 0.1  

ERY_MR = ['GATA1', 'KLF1', 'TAL1', 'LMO2', 'NFE2']
MYE_MR = ['CEBPA', 'SPI1', 'CEBPB', 'CEBPE', 'GFI1',
          'IRF4', 'IRF8', 'BATF3', 'KLF4', 'MAFB']

ERY_LABELS: dict[str, str] = {}
MYE_LABELS: dict[str, str] = {}

NO_CLUSTER_AGGREGATION = {'scenic_regulon'}

CONDITION_ORDER = [
    'tfmodisco_motif',
    'gimme_attribution',
    'gimme_re2g',
    'gimme_abc',
    'gimme_tss_1kbp',
    'scenic_regulon',
]
CONDITION_LABELS = {
    'tfmodisco_motif':   'GenPerturb\n(TF-MoDISco)',
    'gimme_attribution': 'GenPerturb\n(GimmeMotifs)',
    'gimme_re2g':        'rE2G\n(GimmeMotifs)',
    'gimme_abc':         'ABC score\n(GimmeMotifs)',
    'gimme_tss_1kbp':    'TSS ±1 kbp\n(GimmeMotifs)',
    'scenic_regulon':    'SCENIC regulon\n(AUCell)',
}
CONDITION_BAR_COLOR = {
    'tfmodisco_motif':   '#7a3a87',
    'gimme_attribution': '#3b6db8',
    'gimme_re2g':        '#5891c9',
    'gimme_abc':         '#82b3d6',
    'gimme_tss_1kbp':    '#a6c8d8',
    'scenic_regulon':    '#d97c4a',
}

# Lineage → signature column value
LINEAGE_SIGNATURE = {'Erythroid': 'Erythroid', 'Myeloid': 'Granulocyte'}

# Significance / rank tiers
PVAL_THRESHOLD = 0.05
RANK_BINS = [(1, 10), (11, 25), (26, 50), (51, 100)]
RANK_BIN_LABELS = ['Top 1–10', 'Top 11–25', 'Top 26–50', 'Top 51–100']
RANK_BIN_COLORS = ['#67000d', '#cb181d', '#fb6a4a', '#fdc7b1']
SIG_RANK_THRESHOLD = 100  # max rank counted as "significant" tier (for color)


def load_jaspar_cluster_map() -> dict[str, str]:
    df = pd.read_csv(JASPAR_CLUSTERS, sep='\t', usecols=['cluster', 'name'])
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        cluster = str(row['cluster'])
        for entry in str(row['name']).split(','):
            for tf in entry.split('::'):
                tf = tf.strip().upper()
                if tf:
                    out.setdefault(tf, cluster)
    return out


_TF_TO_JASPAR_CLUSTER: dict[str, str] | None = None


def tf_to_cluster_map() -> dict[str, str]:
    global _TF_TO_JASPAR_CLUSTER
    if _TF_TO_JASPAR_CLUSTER is None:
        _TF_TO_JASPAR_CLUSTER = load_jaspar_cluster_map()
    return _TF_TO_JASPAR_CLUSTER


def assign_cluster(tf: str, cond: str) -> str:
    if cond in NO_CLUSTER_AGGREGATION:
        return tf
    return tf_to_cluster_map().get(tf.upper(), tf)


def parse_gimme_motif(motif_id: str) -> list[str]:
    last = motif_id.rsplit('.', 1)[-1]
    parts = re.split(r'::', last)
    return [p.strip().upper() for p in parts if p.strip()]


def parse_modisco_row(row_name: str) -> list[str]:
    parts = re.split(r'::', str(row_name))
    return [p.strip().upper() for p in parts if p.strip()]


def parse_scenic_tf(tf_name: str) -> list[str]:
    return [str(tf_name).replace('(+)', '').strip().upper()]


def _load_signature_real() -> pd.DataFrame:
    df = pd.read_csv(SIGNATURE_FILE, sep='\t')
    df = df[df['value_type'] == 'real'].set_index('Perturbation')
    df = df.drop(columns=['value_type'])
    return df.loc[df.index != 'NT']


def _pearson_rows(motif_df: pd.DataFrame, sig_df: pd.DataFrame,
                  source_name: str, parser, *,
                  min_nonzero: int = 0) -> list[dict]:
    from scipy.stats import pearsonr  # local import to avoid hard dep at top

    rows: list[dict] = []
    for sig_name in LINEAGE_SIGNATURE.values():
        sig_vec = sig_df[sig_name].values.astype(float)
        for motif_name, vec in motif_df.iterrows():
            v = vec.values.astype(float)
            mask = (~np.isnan(v)) & (~np.isnan(sig_vec))
            n_nz = int(np.sum(v[mask] != 0))
            if (mask.sum() < 3 or n_nz < min_nonzero
                    or np.std(v[mask]) == 0 or np.std(sig_vec[mask]) == 0):
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(v[mask], sig_vec[mask])
            rows.append({'signature': sig_name, 'motif': motif_name,
                         'pearson_r': r, 'pearson_pvalue': p,
                         'direction': '',
                         'source': source_name,
                         '_parsed_tfs': parser(motif_name)})
    return rows


def _pearson_rows_pos_neg(motif_pos: pd.DataFrame, motif_neg: pd.DataFrame,
                          sig_df: pd.DataFrame, source_name: str, parser,
                          *, min_nonzero: int = 0) -> list[dict]:
    from scipy.stats import pearsonr  # local import to avoid hard dep at top

    rows: list[dict] = []
    motifs = motif_pos.index.union(motif_neg.index)
    for sig_name in LINEAGE_SIGNATURE.values():
        sig_vec = sig_df[sig_name].values.astype(float)
        for motif_name in motifs:
            cands: list[tuple[str, float, float]] = []
            for direction, mat in (('pos', motif_pos), ('neg', motif_neg)):
                if motif_name not in mat.index:
                    continue
                v = mat.loc[motif_name].values.astype(float)
                mask = (~np.isnan(v)) & (~np.isnan(sig_vec))
                n_nz = int(np.sum(v[mask] != 0))
                if (mask.sum() < 3 or n_nz < min_nonzero
                        or np.std(v[mask]) == 0
                        or np.std(sig_vec[mask]) == 0):
                    continue
                r, p = pearsonr(v[mask], sig_vec[mask])
                cands.append((direction, r, p))
            if not cands:
                rows.append({'signature': sig_name, 'motif': motif_name,
                             'pearson_r': np.nan, 'pearson_pvalue': np.nan,
                             'direction': '',
                             'source': source_name,
                             '_parsed_tfs': parser(motif_name)})
                continue
            direction, r, p = max(cands, key=lambda x: abs(x[1]))
            rows.append({'signature': sig_name, 'motif': motif_name,
                         'pearson_r': r, 'pearson_pvalue': p,
                         'direction': direction,
                         'source': source_name,
                         '_parsed_tfs': parser(motif_name)})
    return rows


def _compute_tfmodisco_pearson() -> pd.DataFrame:
    motif_pos = pd.read_csv(TFMODISCO_MATRIX_POS, sep='\t', index_col=0)
    motif_neg = pd.read_csv(TFMODISCO_MATRIX_NEG, sep='\t', index_col=0)
    motif_pos.columns = [c.replace('Norman.', '') for c in motif_pos.columns]
    motif_neg.columns = [c.replace('Norman.', '') for c in motif_neg.columns]
    sig_df = _load_signature_real()
    common = (motif_pos.columns
              .intersection(motif_neg.columns)
              .intersection(sig_df.index))
    return pd.DataFrame(_pearson_rows_pos_neg(
        motif_pos[common], motif_neg[common], sig_df.loc[common],
        source_name='tfmodisco_motif', parser=parse_modisco_row))


def _compute_scenic_pearson() -> pd.DataFrame:
    auc = pd.read_csv(AUCELL_FILE, sep='\t', index_col=0)
    auc.columns = [c.split('(')[0].strip() for c in auc.columns]
    if 'control' in auc.index:
        auc = auc.rename(index={'control': 'NT'})
    auc = auc.loc[auc.index != 'NT']

    sig_df = _load_signature_real()
    common = auc.index.intersection(sig_df.index)
    tf_df = auc.loc[common].T
    min_nz = max(1, int(len(common) * MIN_NONZERO_FRAC))
    return pd.DataFrame(_pearson_rows(
        tf_df, sig_df.loc[common],
        source_name='scenic_regulon', parser=parse_scenic_tf,
        min_nonzero=min_nz))


def _build_gimme_pvalue_matrix(bed_type: str) -> pd.DataFrame:
    if bed_type == 'attribution':
        existing = GIMME_BASE / 'gimme_motif_pvalue_matrix.tsv'
        if existing.exists():
            return pd.read_csv(existing, sep='\t', index_col=0)

    results: dict[str, pd.Series] = {}
    for pert_dir in sorted(p for p in GIMME_BASE.iterdir() if p.is_dir()):
        if bed_type == 'attribution':
            report = pert_dir / 'gimme.roc.report.txt'
        else:
            report = pert_dir / bed_type / 'gimme.roc.report.txt'
        if not report.exists():
            continue
        try:
            rep = pd.read_csv(report, sep='\t')
            results[pert_dir.name] = rep.set_index('Motif')['log10 P-value']
        except Exception as e:
            print(f'  [WARN] Failed to parse {report}: {e}')
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).fillna(0)


def _expressed_genes_uppercase() -> set[str]:
    df = pd.read_csv(EXPRESSION_FILE, sep='\t', index_col=0)
    df = df.drop(columns=['training'], errors='ignore')
    mean_expr = df.mean(axis=1)
    return {str(g).upper() for g in mean_expr[mean_expr >= EXPRESSION_MIN_MEAN].index}


def _jaspar_motif_tf_name(motif_id: str) -> str:
    if '_' in motif_id:
        after = motif_id.split('_', 1)[1]
        parts = after.split('.', 2)
        if len(parts) >= 3:
            return parts[2]
    return motif_id


def _filter_gimme_motifs_by_expression(matrix: pd.DataFrame,
                                       expressed: set[str]) -> pd.DataFrame:
    keep = []
    for motif in matrix.index:
        tf = _jaspar_motif_tf_name(motif)
        genes = [g.strip().upper() for g in tf.split('::')]
        keep.append(any(g in expressed for g in genes))
    return matrix.loc[keep]


def _compute_gimme_pearson(cond: str, bed_type: str,
                           expressed: set[str]) -> pd.DataFrame:
    matrix = _build_gimme_pvalue_matrix(bed_type)
    if matrix.empty:
        print(f'  [WARN] no gimme reports for {cond} ({bed_type})')
        return pd.DataFrame()

    matrix.columns = [c.replace('Norman.', '') for c in matrix.columns]
    matrix = _filter_gimme_motifs_by_expression(matrix, expressed)

    sig_df = _load_signature_real()
    common = matrix.columns.intersection(sig_df.index)
    if len(common) == 0:
        return pd.DataFrame()

    min_nz = max(1, int(len(common) * MIN_NONZERO_FRAC))
    return pd.DataFrame(_pearson_rows(
        matrix[common], sig_df.loc[common],
        source_name=cond, parser=parse_gimme_motif,
        min_nonzero=min_nz))


def load_all_correlations() -> dict[str, pd.DataFrame]:
    by_cond: dict[str, pd.DataFrame] = {}

    print('[compute] tfmodisco — Pearson from raw motif matrix...')
    by_cond['tfmodisco_motif'] = _explode_to_tf(_compute_tfmodisco_pearson())

    print('[compute] gimme — Pearson from gimme.roc.report per CRE source...')
    expressed = _expressed_genes_uppercase()
    print(f'  expressed-gene filter: {len(expressed)} genes (mean ≥ {EXPRESSION_MIN_MEAN})')
    for cond, bed_type in GIMME_BED_TYPES.items():
        df = _compute_gimme_pearson(cond, bed_type, expressed)
        if df.empty:
            print(f'  [skip] {cond}: no rows')
            continue
        by_cond[cond] = _explode_to_tf(df)

    print('[compute] scenic — Pearson from AUCell mean per perturbation...')
    by_cond['scenic_regulon'] = _explode_to_tf(_compute_scenic_pearson())

    return by_cond


def _explode_to_tf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.explode('_parsed_tfs').rename(columns={'_parsed_tfs': 'tf'})
    df = df.dropna(subset=['tf'])
    df['abs_pearson_r'] = df['pearson_r'].abs()

    df = df.sort_values(['signature', 'tf', 'abs_pearson_r'],
                        ascending=[True, True, False])
    best = df.groupby(['signature', 'tf'], as_index=False).first()
    min_p = df.groupby(['signature', 'tf'])['pearson_pvalue'].min().reset_index()
    min_p = min_p.rename(columns={'pearson_pvalue': 'min_pearson_pvalue'})
    n_motifs = (df.dropna(subset=['pearson_r'])
                  .groupby(['signature', 'tf'])['motif']
                  .nunique()
                  .reset_index()
                  .rename(columns={'motif': 'n_source_motifs'}))
    out = best.merge(min_p, on=['signature', 'tf'])
    out = out.merge(n_motifs, on=['signature', 'tf'], how='left')
    out['n_source_motifs'] = out['n_source_motifs'].fillna(0).astype(int)
    out = out.rename(columns={'motif': 'source_motif',
                              'pearson_pvalue': 'pearson_pvalue_at_max_abs_r'})
    if 'direction' not in out.columns:
        out['direction'] = ''
    return out[['signature', 'tf', 'pearson_r', 'abs_pearson_r',
                'min_pearson_pvalue', 'pearson_pvalue_at_max_abs_r',
                'direction', 'source_motif', 'n_source_motifs', 'source']]


def gene_level_table(cond_dfs: dict[str, pd.DataFrame], lineage: str
                    ) -> pd.DataFrame:
    sig = LINEAGE_SIGNATURE[lineage]
    rows = []
    for cond, df in cond_dfs.items():
        sub = df[df['signature'] == sig].copy()
        sub = sub.sort_values('abs_pearson_r', ascending=False).reset_index(drop=True)
        sub['rank'] = np.arange(1, len(sub) + 1)
        sub['condition'] = cond
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def cluster_level_table(cond_dfs: dict[str, pd.DataFrame], lineage: str
                       ) -> pd.DataFrame:
    sig = LINEAGE_SIGNATURE[lineage]
    rows = []
    for cond, df in cond_dfs.items():
        sub = df[df['signature'] == sig].copy()
        sub['family'] = sub['tf'].map(lambda t: assign_cluster(t, cond))
        # Aggregate per cluster: row with max |Pearson r|
        sub = sub.sort_values('abs_pearson_r', ascending=False)
        fam_best = sub.groupby('family', as_index=False).first()
        # min Pearson p across cluster members
        fam_min_p = sub.groupby('family', as_index=False)['min_pearson_pvalue'].min()
        fam_best = fam_best.drop(columns=['min_pearson_pvalue']).merge(
            fam_min_p, on='family')
        # Cluster richness: # TFs collapsed and # source motifs across them.
        fam_n_tfs = (sub.groupby('family', as_index=False)['tf']
                       .nunique()
                       .rename(columns={'tf': 'n_source_tfs'}))
        fam_best = fam_best.drop(columns=['n_source_motifs'], errors='ignore')
        if 'n_source_motifs' in sub.columns:
            fam_n_motifs = (sub.groupby('family', as_index=False)
                                ['n_source_motifs'].sum())
            fam_best = fam_best.merge(fam_n_motifs, on='family')
        fam_best = fam_best.merge(fam_n_tfs, on='family')
        fam_best = fam_best.sort_values('abs_pearson_r', ascending=False).reset_index(drop=True)
        fam_best['rank'] = np.arange(1, len(fam_best) + 1)
        fam_best['condition'] = cond
        rows.append(fam_best)
    return pd.concat(rows, ignore_index=True)


def lookup_master_regulators(panel_table: pd.DataFrame,
                             mr_list: list[str], level: str,
                             ) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        sub = panel_table[panel_table['condition'] == cond]
        for mr in mr_list:
            if level == 'gene':
                key = mr
            else:
                key = assign_cluster(mr, cond)
            if level == 'gene':
                hit = sub[sub['tf'] == key]
            else:
                hit = sub[sub['family'] == key]
            if len(hit) == 0:
                rows.append({'master_regulator': mr, 'condition': cond,
                             'level': level, 'lookup_key': key,
                             'pearson_r': np.nan, 'abs_pearson_r': np.nan,
                             'min_pearson_pvalue': np.nan,
                             'pearson_pvalue_at_max_abs_r': np.nan,
                             'rank': np.nan,
                             'is_significant': False, 'tier': None,
                             'direction': '',
                             'source_motif': '',
                             'n_source_motifs': 0,
                             'n_source_tfs': 0})
                continue
            r = hit.iloc[0]
            sig_p = (r['min_pearson_pvalue'] < PVAL_THRESHOLD) if pd.notna(r['min_pearson_pvalue']) else False
            tier = tier_for_rank(int(r['rank'])) if pd.notna(r['rank']) else None
            rows.append({'master_regulator': mr, 'condition': cond,
                         'level': level, 'lookup_key': key,
                         'pearson_r': float(r['pearson_r']),
                         'abs_pearson_r': float(r['abs_pearson_r']),
                         'min_pearson_pvalue': float(r['min_pearson_pvalue']),
                         'pearson_pvalue_at_max_abs_r': float(r['pearson_pvalue_at_max_abs_r']),
                         'rank': int(r['rank']),
                         'is_significant': bool(sig_p),
                         'tier': tier,
                         'direction': str(r.get('direction', '')),
                         'source_motif': str(r.get('source_motif', '')),
                         'n_source_motifs': int(r.get('n_source_motifs', 0) or 0),
                         'n_source_tfs': int(r.get('n_source_tfs', 0) or 0)})
    return pd.DataFrame(rows)


def tier_for_rank(rank: int | None) -> int | None:
    if rank is None or rank > SIG_RANK_THRESHOLD:
        return None
    for i, (lo, hi) in enumerate(RANK_BINS):
        if lo <= rank <= hi:
            return i
    return None


def plot_upset(table: pd.DataFrame, mr_list: list[str], lineage: str, level: str,
               display_labels: dict[str, str], outpath_prefix: Path,
               universe_counts: dict[str, int]):
    n_cond = len(CONDITION_ORDER)
    n_mr = len(mr_list)

    # Pivot to matrices
    tier_mat = np.full((n_mr, n_cond), np.nan)
    rank_mat = np.full((n_mr, n_cond), np.nan)
    sig_mat = np.zeros((n_mr, n_cond), dtype=bool)
    pearson_r_mat = np.full((n_mr, n_cond), np.nan)
    for _, row in table.iterrows():
        i = mr_list.index(row['master_regulator'])
        j = CONDITION_ORDER.index(row['condition'])
        if pd.notna(row['rank']):
            rank_mat[i, j] = row['rank']
        if pd.notna(row['pearson_r']):
            pearson_r_mat[i, j] = row['pearson_r']
        sig_mat[i, j] = bool(row['is_significant'])
        if row['tier'] is not None and not (
                isinstance(row['tier'], float) and np.isnan(row['tier'])):
            tier_mat[i, j] = row['tier']

    bar_counts = np.array([universe_counts[c] for c in CONDITION_ORDER], dtype=int)
    bar_unit = 'genes' if level == 'gene' else 'clusters'
    bar_label = f'# sig. {bar_unit}\n(Pearson p < 0.05)'

    fig_w = 1.6 * n_cond + 5.5
    fig_h = 0.6 * n_mr + 5.0
    fig = plt.figure(figsize=(max(fig_w, 11), max(fig_h, 7.5)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.5 + 0.35 * n_mr],
                          hspace=0.08, top=0.88, bottom=0.18,
                          left=0.18, right=0.78)
    ax_bar = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)

    x = np.arange(n_cond)
    bars = ax_bar.bar(x, bar_counts, width=0.7,
                      color=[CONDITION_BAR_COLOR[c] for c in CONDITION_ORDER],
                      edgecolor='black', linewidth=0.8)
    for b, v in zip(bars, bar_counts):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.05,
                    f'{int(v)}', ha='center', va='bottom', fontsize=14)
    ax_bar.set_ylabel(bar_label, fontsize=15)
    max_bar = int(max(bar_counts.max() if bar_counts.size else 1, 1))
    ax_bar.set_ylim(0, max_bar * 1.18 + 1.0)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(labelbottom=False, length=0)

    fig.suptitle(
        f'{lineage} master regulators — {level}-level UpSet '
        f'(MR genes = {n_mr}, signature = {LINEAGE_SIGNATURE[lineage]}, '
        f'ranking = Pearson correlation coefficient)',
        fontsize=18, y=0.97)

    for i in range(n_mr):
        ax_dot.axhspan(i - 0.5, i + 0.5,
                       color='#f5f5f5' if i % 2 == 0 else 'white',
                       zorder=0)

    for j in range(n_cond):
        sig_rows = [i for i in range(n_mr) if sig_mat[i, j]
                    and not np.isnan(tier_mat[i, j])]
        if len(sig_rows) >= 2:
            ax_dot.plot([j] * len(sig_rows), sig_rows,
                        color='#444444', linewidth=1.5, alpha=0.55,
                        zorder=2)

    for i in range(n_mr):
        for j in range(n_cond):
            tier = tier_mat[i, j]
            rank = rank_mat[i, j]
            sig = sig_mat[i, j]

            if np.isnan(rank):
                continue 

            if not sig:
                ax_dot.scatter(j, i, s=40, marker='o',
                               color='#d9d9d9', edgecolor='none', zorder=3)
                continue

            if np.isnan(tier):
                ax_dot.scatter(j, i, s=320, marker='o',
                               facecolor='white', edgecolor='#888888',
                               linewidth=0.8, zorder=4)
                ax_dot.text(j, i, f'{int(rank)}',
                            ha='center', va='center', fontsize=8,
                            color='black', zorder=5)
                continue

            color = RANK_BIN_COLORS[int(tier)]
            ax_dot.scatter(j, i, s=460, marker='o', color=color,
                           edgecolor='none', zorder=4)
            ax_dot.text(j, i, f'{int(rank)}',
                        ha='center', va='center', fontsize=11,
                        color='white' if int(tier) < 2 else 'black',
                        zorder=5)

    ax_dot.set_xlim(-0.5, n_cond - 0.5)
    ax_dot.set_ylim(-0.5, n_mr - 0.5)
    ax_dot.set_xticks(x)
    ax_dot.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER],
                           rotation=0, fontsize=13)
    ax_dot.set_yticks(np.arange(n_mr))
    y_labels = [display_labels.get(m, m) for m in mr_list]
    ax_dot.set_yticklabels(y_labels, fontsize=16)
    ax_dot.invert_yaxis()
    ax_dot.set_xlabel('Evidence source', fontsize=17, labelpad=10)
    ax_dot.set_ylabel(f'Master regulator ({lineage})', fontsize=17)
    ax_dot.spines['top'].set_visible(False)
    ax_dot.spines['right'].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=RANK_BIN_COLORS[i], edgecolor='none',
                       label=RANK_BIN_LABELS[i])
        for i in range(len(RANK_BINS))
    ]
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='#888888',
                                         label=f'Sig. Pearson p<0.05,\nrank > {SIG_RANK_THRESHOLD}'))
    legend_handles.append(mpatches.Patch(facecolor='#d9d9d9', edgecolor='none',
                                         label='Not significant'))
    ax_dot.legend(handles=legend_handles, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12,
                  title='Rank within condition\n(by Pearson correlation\ncoefficient)',
                  title_fontsize=12)

    fig.savefig(f'{outpath_prefix}.svg', bbox_inches='tight')
    fig.savefig(f'{outpath_prefix}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {outpath_prefix}.{{svg,png}}')


def universe_significant_counts(panel_table: pd.DataFrame, level: str
                               ) -> dict[str, int]:
    out: dict[str, int] = {}
    sig = panel_table[panel_table['min_pearson_pvalue'] < PVAL_THRESHOLD]
    if level == 'gene':
        for cond in CONDITION_ORDER:
            out[cond] = int(sig[sig['condition'] == cond]['tf'].nunique())
    else:
        for cond in CONDITION_ORDER:
            out[cond] = int(sig[sig['condition'] == cond]['family'].nunique())
    return out


def main():
    cond_dfs = load_all_correlations()

    long_rows = []
    for cond, df in cond_dfs.items():
        df2 = df.copy()
        df2['condition'] = cond
        long_rows.append(df2)
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(OUTPUT_DIR / 'tf_correlation_long.tsv', sep='\t', index=False)
    print(f'[save] {OUTPUT_DIR / "tf_correlation_long.tsv"} '
          f'({len(long_df)} rows)')

    panels = [
        ('Erythroid', 'gene',    ERY_MR, ERY_LABELS),
        ('Erythroid', 'cluster', ERY_MR, ERY_LABELS),
        ('Myeloid',   'gene',    MYE_MR, MYE_LABELS),
        ('Myeloid',   'cluster', MYE_MR, MYE_LABELS),
    ]

    universe_summary_rows = []

    for lineage, level, mr_list, labels in panels:
        if level == 'gene':
            panel = gene_level_table(cond_dfs, lineage)
        else:
            panel = cluster_level_table(cond_dfs, lineage)

        uni_counts = universe_significant_counts(panel, level)
        uni_total = {}
        for cond in CONDITION_ORDER:
            sub = panel[panel['condition'] == cond]
            key = 'tf' if level == 'gene' else 'family'
            uni_total[cond] = int(sub[key].nunique())
        for cond in CONDITION_ORDER:
            universe_summary_rows.append({
                'lineage': lineage, 'level': level, 'condition': cond,
                'signature': LINEAGE_SIGNATURE[lineage],
                'universe_size': uni_total[cond],
                'n_significant': uni_counts[cond],
            })

        mr_table = lookup_master_regulators(panel, mr_list, level)
        out_tsv = OUTPUT_DIR / f'master_regulator_table_{lineage.lower()}_{level}.tsv'
        mr_table.to_csv(out_tsv, sep='\t', index=False)
        print(f'[save] {out_tsv}')

        prefix = FIGURE_DIR / f'upset_{lineage.lower()}_{level}'
        plot_upset(mr_table, mr_list, lineage, level, labels, prefix, uni_counts)

    pd.DataFrame(universe_summary_rows).to_csv(
        OUTPUT_DIR / 'condition_universe_summary.tsv', sep='\t', index=False)
    print(f'[save] {OUTPUT_DIR / "condition_universe_summary.tsv"}')


if __name__ == '__main__':
    main()
