#!/usr/bin/env python
import os
import sys
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gene_aware_intervals import (  # noqa: E402
    assign_max_overlap_scores_by_gene,
    build_interval_index_by_gene,
    iter_overlapping_intervals,
)


def _as_text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_attribution_bedgraph(bedgraph_path: str) -> pd.DataFrame:
    if not os.path.exists(bedgraph_path):
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score', 'gene', 'pert'])

    df = pd.read_csv(bedgraph_path, sep='\t', header=None, comment='#')
    if len(df) == 0 or df.shape[1] < 4:
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score', 'gene', 'pert'])

    result = pd.DataFrame({
        'chr': df.iloc[:, 0].astype(str),
        'start': pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).astype(int),
        'end': pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(0).astype(int),
        'score': pd.to_numeric(df.iloc[:, 3], errors='coerce').fillna(0.0),
    })

    if df.shape[1] >= 5:
        result['gene'] = df.iloc[:, 4].astype(str)
    else:
        result['gene'] = 'unknown'

    if df.shape[1] >= 6:
        result['pert'] = df.iloc[:, 5].astype(str)
    else:
        result['pert'] = 'unknown'

    result = result[(result['start'] >= 0) & (result['end'] > result['start'])]

    return result


def load_unified_peaks(bed_path: str) -> pd.DataFrame:
    if not os.path.exists(bed_path):
        return pd.DataFrame()

    df = pd.read_csv(bed_path, sep='\t', header=None, comment='#')
    if len(df) == 0:
        return pd.DataFrame()

    cols = ['chr', 'start', 'end', 'gene', 'sources', 'max_score', 'atac_overlap']
    result = pd.DataFrame()

    for i, col in enumerate(cols):
        if i < df.shape[1]:
            result[col] = df.iloc[:, i]

    if 'start' in result.columns:
        result['start'] = pd.to_numeric(result['start'], errors='coerce').fillna(0).astype(int)
    if 'end' in result.columns:
        result['end'] = pd.to_numeric(result['end'], errors='coerce').fillna(0).astype(int)
    if 'max_score' in result.columns:
        result['max_score'] = pd.to_numeric(result['max_score'], errors='coerce').fillna(0.0)
    if 'atac_overlap' in result.columns:
        result['atac_overlap'] = pd.to_numeric(result['atac_overlap'], errors='coerce').fillna(0).astype(int)

    return result


def compute_peak_attribution_scores(peaks_df: pd.DataFrame, bedgraph_df: pd.DataFrame) -> pd.DataFrame:
    peaks_df = peaks_df.copy()

    if len(peaks_df) == 0:
        peaks_df['attr_top10_score'] = pd.Series(dtype=float)
        return peaks_df

    if len(bedgraph_df) == 0:
        peaks_df['attr_top10_score'] = 0.0
        return peaks_df

    bg_by_gene = build_interval_index_by_gene(bedgraph_df)

    def compute_top10_score(row):
        peak_start = row['start']
        peak_end = row['end']
        intervals = bg_by_gene.get((str(row['chr']), str(row['gene'])), [])
        total_bp = 0
        abs_values = []  # (abs_score, bp) pairs for top10 calculation

        for bg_start, bg_end, bg_score in iter_overlapping_intervals(
            intervals, peak_start, peak_end
        ):
            overlap_start = max(peak_start, bg_start)
            overlap_end = min(peak_end, bg_end)
            overlap_bp = overlap_end - overlap_start

            if overlap_bp > 0:
                total_bp += overlap_bp
                abs_values.append((abs(bg_score), overlap_bp))

        if total_bp == 0 or len(abs_values) == 0:
            return 0.0

        abs_values.sort(key=lambda x: x[0], reverse=True)
        top10_bp_target = max(1, int(total_bp * 0.1))
        top10_score = 0.0
        top10_bp = 0
        for val, bp in abs_values:
            take_bp = min(bp, top10_bp_target - top10_bp)
            top10_score += val * take_bp
            top10_bp += take_bp
            if top10_bp >= top10_bp_target:
                break
        return top10_score / top10_bp if top10_bp > 0 else 0.0

    peaks_df['attr_top10_score'] = peaks_df.apply(compute_top10_score, axis=1)

    return peaks_df


def compute_peak_attribution_scores_from_raw_h5(
    peaks_df: pd.DataFrame,
    raw_h5_path: str,
    top_fraction: float = 0.1,
) -> pd.DataFrame:
    if not 0 < top_fraction <= 1:
        raise ValueError(f'top_fraction must be in (0, 1], got {top_fraction}')
    if not os.path.exists(raw_h5_path):
        raise FileNotFoundError(f'Missing raw attribution H5: {raw_h5_path}')

    peaks_df = peaks_df.copy()
    peaks_df['attr_top10_score'] = 0.0
    if len(peaks_df) == 0:
        return peaks_df

    with h5py.File(raw_h5_path, 'r') as hf:
        for gene, peak_indexes in peaks_df.groupby('gene').groups.items():
            gene = str(gene)
            if gene not in hf or 'ixg_fc' not in hf[gene]:
                continue

            grp = hf[gene]
            chromosome = _as_text(grp.attrs.get('chromosome', ''))
            seq_start = int(grp.attrs.get('seq_start', 0))
            ixg_fc = grp['ixg_fc']
            seq_end = seq_start + int(ixg_fc.shape[0])

            for peak_idx in peak_indexes:
                peak = peaks_df.loc[peak_idx]
                if str(peak['chr']) != chromosome:
                    continue

                genomic_start = max(int(peak['start']), seq_start)
                genomic_end = min(int(peak['end']), seq_end)
                if genomic_start >= genomic_end:
                    continue

                rel_start = genomic_start - seq_start
                rel_end = genomic_end - seq_start
                per_base = np.abs(ixg_fc[rel_start:rel_end, :]).sum(axis=1)
                n_top = max(1, int(np.ceil(len(per_base) * top_fraction)))
                if n_top < len(per_base):
                    per_base = np.partition(per_base, len(per_base) - n_top)[-n_top:]
                peaks_df.at[peak_idx, 'attr_top10_score'] = float(per_base.mean())

    return peaks_df


def load_tss_bed(tss_bed_path: str) -> pd.DataFrame:
    if not os.path.exists(tss_bed_path):
        return pd.DataFrame(columns=['chr', 'tss_pos', 'gene'])

    df = pd.read_csv(tss_bed_path, sep='\t', header=None, comment='#',
                     names=['chr', 'start', 'end', 'gene', 'score', 'strand', 'split'])

    df['tss_pos'] = np.where(df['strand'] == '+', df['start'], df['end'])

    return df[['chr', 'tss_pos', 'gene']]


def compute_tss_distance_score(peaks_df: pd.DataFrame, tss_df: pd.DataFrame) -> pd.DataFrame:
    peaks_df = peaks_df.copy()

    if len(peaks_df) == 0 or len(tss_df) == 0:
        peaks_df['tss_distance_score'] = 0.0
        return peaks_df

    gene_tss = {}
    for _, row in tss_df.iterrows():
        gene_tss[row['gene']] = (row['chr'], int(row['tss_pos']))

    def calc_distance_and_score(row):
        gene = row['gene']
        if gene not in gene_tss:
            return 0.0, np.nan
        tss_chr, tss_pos = gene_tss[gene]
        if str(row['chr']) != str(tss_chr):
            return 0.0, np.nan
        peak_mid = (row['start'] + row['end']) / 2.0
        distance = abs(peak_mid - tss_pos)
        score = 1.0 / (1.0 + distance)
        return score, distance

    results = peaks_df.apply(calc_distance_and_score, axis=1, result_type='expand')
    peaks_df['tss_distance_score'] = results[0]
    peaks_df['tss_distance_bp'] = results[1]

    return peaks_df


def load_source_scores(cre_dir: str, pert: str, peaks_df: pd.DataFrame) -> pd.DataFrame:
    peaks_df = peaks_df.copy()
    peaks_df['abc_score'] = 0.0
    peaks_df['re2g_score'] = 0.0
    peaks_df['re2g_ext_score'] = 0.0

    if len(peaks_df) == 0:
        return peaks_df

    def assign_scores_from_source(peaks_df, source_bed, score_col, gene_col, target_col):
        if not os.path.exists(source_bed):
            return peaks_df

        source_df = pd.read_csv(source_bed, sep='\t', header=None, comment='#')
        if len(source_df) == 0 or source_df.shape[1] <= max(score_col, gene_col):
            return peaks_df

        source_bt_df = pd.DataFrame({
            'chr': source_df.iloc[:, 0].astype(str),
            'start': pd.to_numeric(source_df.iloc[:, 1], errors='coerce').fillna(0).astype(int),
            'end': pd.to_numeric(source_df.iloc[:, 2], errors='coerce').fillna(0).astype(int),
            'gene': source_df.iloc[:, gene_col].astype(str),
            'score': pd.to_numeric(source_df.iloc[:, score_col], errors='coerce').fillna(0.0),
        })
        source_bt_df = source_bt_df[(source_bt_df['start'] >= 0) & (source_bt_df['end'] > source_bt_df['start'])]

        if len(source_bt_df) == 0:
            return peaks_df

        try:
            peaks_df[target_col] = assign_max_overlap_scores_by_gene(peaks_df, source_bt_df)
        except Exception as e:
            print(f'    [WARN] Error assigning scores: {e}')

        return peaks_df

    abc_bed = os.path.join(cre_dir, f'abc_score_{pert}.bed')
    peaks_df = assign_scores_from_source(peaks_df, abc_bed, score_col=4, gene_col=3, target_col='abc_score')

    re2g_bed = os.path.join(cre_dir, f're2g_{pert}.bed')
    peaks_df = assign_scores_from_source(peaks_df, re2g_bed, score_col=4, gene_col=6, target_col='re2g_score')

    re2g_ext_bed = os.path.join(cre_dir, f're2g_extended_{pert}.bed')
    peaks_df = assign_scores_from_source(peaks_df, re2g_ext_bed, score_col=4, gene_col=6, target_col='re2g_ext_score')

    return peaks_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study_name', default='NormanWeissman2019_filtered_mixscape_exnp_train')
    ap.add_argument('--study_suffix', default='alphagenome_transfer_epoch100_batch256_adamw5e3')

    ap.add_argument('--attribution_root', default='attribution')
    ap.add_argument('--cre_root', default='cre')
    ap.add_argument('--output_root', default='cre')

    ap.add_argument('--tfs', default='', help='Comma-separated TF list. Empty => infer from unified_peaks dir')
    ap.add_argument('--tss_bed', default='', help='Path to TSS BED file. Default: fasta/{study_name}.bed')
    ap.add_argument('--attribution_filename_suffix', default='',
                    help="Use {pert}{suffix}_attribution_fc.bedgraph or {pert}{suffix}_raw_attribution.h5 (e.g. '_gtgenes')")
    ap.add_argument('--attribution_score_source', choices=['bedgraph', 'raw_h5'], default='bedgraph',
                    help="Compute attr_top10_score from 128-bp BedGraph bins (legacy default) or raw per-base ixg_fc")

    args = ap.parse_args()

    study = f'{args.study_name}__{args.study_suffix}'

    tss_bed_path = args.tss_bed if args.tss_bed.strip() else f'fasta/{args.study_name}.bed'
    tss_df = load_tss_bed(tss_bed_path)
    if len(tss_df) > 0:
        print(f'[INFO] Loaded {len(tss_df)} TSS entries from {tss_bed_path}')
    else:
        print(f'[WARN] No TSS data found at {tss_bed_path}. TSS distance score will be 0.')

    unified_peaks_dir = os.path.join(args.cre_root, study, 'unified_peaks')
    output_dir = os.path.join(args.output_root, study, 'scored_peaks')
    os.makedirs(output_dir, exist_ok=True)

    if args.tfs.strip():
        tfs = [x.strip() for x in args.tfs.split(',') if x.strip()]
    else:
        if not os.path.isdir(unified_peaks_dir):
            raise FileNotFoundError(f'[ERROR] Missing directory: {unified_peaks_dir}')

        tfs = []
        for f in os.listdir(unified_peaks_dir):
            if f.startswith('unified_peaks_') and f.endswith('.bed'):
                pert = f.replace('unified_peaks_', '').replace('.bed', '')
                tfs.append(pert)
        tfs = sorted(tfs)

    if len(tfs) == 0:
        raise ValueError('[ERROR] TF list is empty.')

    print(f'[INFO] Found {len(tfs)} TFs')

    summary_rows = []

    for pert in tfs:
        tf_symbol = pert.split('.')[-1]
        print(f'\n[INFO] Processing {pert} (TF={tf_symbol})...')

        unified_bed = os.path.join(unified_peaks_dir, f'unified_peaks_{pert}.bed')
        bedgraph_path = os.path.join(args.attribution_root, study, pert,
                                     f'{pert}{args.attribution_filename_suffix}_attribution_fc.bedgraph')
        raw_h5_path = os.path.join(args.attribution_root, study, pert,
                                   f'{pert}{args.attribution_filename_suffix}_raw_attribution.h5')
        cre_pert_dir = os.path.join(args.cre_root, study, pert)

        if not os.path.exists(unified_bed):
            print(f'[WARN] Missing unified peaks: {unified_bed}. Skipping.')
            continue

        peaks_df = load_unified_peaks(unified_bed)
        if len(peaks_df) == 0:
            print(f'[WARN] No peaks in {unified_bed}. Skipping.')
            continue

        print(f'  Loaded {len(peaks_df)} unified peaks')

        if args.attribution_score_source == 'raw_h5':
            peaks_df = compute_peak_attribution_scores_from_raw_h5(peaks_df, raw_h5_path)
            print(f'  Computed attribution scores from raw per-base H5: {raw_h5_path}')
        elif os.path.exists(bedgraph_path):
            bedgraph_df = load_attribution_bedgraph(bedgraph_path)
            print(f'  Loaded {len(bedgraph_df)} bedgraph entries')
            peaks_df = compute_peak_attribution_scores(peaks_df, bedgraph_df)
            print(f'  Computed attribution scores')
        else:
            print(f'  [WARN] Missing bedgraph: {bedgraph_path}')
            peaks_df['attr_top10_score'] = 0.0

        peaks_df = load_source_scores(cre_pert_dir, pert, peaks_df)

        peaks_df = compute_tss_distance_score(peaks_df, tss_df)

        output_bed = os.path.join(output_dir, f'scored_peaks_{pert}.tsv')
        peaks_df.to_csv(output_bed, sep='\t', index=False)
        print(f'  Saved: {output_bed}')

        n_positive = int(peaks_df['atac_overlap'].sum())
        n_negative = len(peaks_df) - n_positive

        summary_row = {
            'study': study,
            'pert': pert,
            'tf_symbol': tf_symbol,
            'n_peaks': len(peaks_df),
            'n_positive': n_positive,
            'n_negative': n_negative,
        }
        for col in peaks_df.columns:
            if col.endswith('_score'):
                summary_row[f'mean_{col}'] = float(peaks_df[col].mean())
        summary_rows.append(summary_row)

        summary_file = os.path.join(output_dir, f'summary_{pert}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_row, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = os.path.join(output_dir, 'scored_peaks_summary.tsv')
    summary_df.to_csv(summary_tsv, sep='\t', index=False)
    print(f'\n[INFO] Saved overall summary: {summary_tsv}')
    print(f'[INFO] Processed {len(summary_rows)} TFs')


if __name__ == '__main__':
    main()
