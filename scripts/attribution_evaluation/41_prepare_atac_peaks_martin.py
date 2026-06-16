#!/usr/bin/env python
import os
import pandas as pd


def main():
    xlsx_path = 'data/science.ads7951_tables_s1_to_s6/science.ads7951_table_s2.xlsx'
    output_dir = 'reference/martin_atac'
    os.makedirs(output_dir, exist_ok=True)

    print(f'[INFO] Reading {xlsx_path}...')
    df = pd.read_excel(xlsx_path, sheet_name='Table_S2_TF_sensitive_ACRs',
                        header=2)  # Row 2 is the header row (0-indexed)

    print(f'[INFO] Total rows: {len(df)}')
    print(f'[INFO] Columns: {list(df.columns)}')

    locus_split = df['peak_locus'].str.split('-', n=2, expand=True)
    df['chr'] = locus_split[0]
    df['start'] = locus_split[1].astype(int)
    df['end'] = locus_split[2].astype(int)

    for pert_name, group in df.groupby('perturbation_name'):
        bed_df = group[['chr', 'start', 'end', 'log2FC', 'beta_weight', 'p_weight']].copy()
        bed_df = bed_df.sort_values(['chr', 'start']).reset_index(drop=True)

        output_path = os.path.join(output_dir, f'{pert_name}.bed')
        bed_df.to_csv(output_path, sep='\t', index=False, header=False)
        print(f'  {pert_name}: {len(bed_df)} peaks -> {output_path}')

    summary = df.groupby('perturbation_name').size().reset_index(name='n_peaks')
    summary_path = os.path.join(output_dir, 'summary.tsv')
    summary.to_csv(summary_path, sep='\t', index=False)
    print(f'\n[INFO] Summary saved to {summary_path}')
    print('[INFO] Done.')


if __name__ == '__main__':
    main()
