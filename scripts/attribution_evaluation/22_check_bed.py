#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bed_utils import (  # noqa: E402
    BED_TRACK_COLORS,
    get_bed_track_configs,
    safe_makedirs,
)



GENE_COLORS = [
    '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
    '#1ABC9C', '#E91E63', '#00BCD4', '#FF5722', '#607D8B',
    '#8BC34A', '#FF9800', '#795548', '#009688', '#673AB7',
    '#CDDC39', '#03A9F4', '#FFC107', '#9C27B0', '#4CAF50',
]

PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


@dataclass
class BedGraphData:
    """Container for bedgraph data."""
    chromosomes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    genes: np.ndarray
    perts: np.ndarray


@dataclass
class BedTrackData:
    """Container for a single BED track."""
    name: str
    chromosomes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    genes: np.ndarray
    color: str
    label: str


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    width: float = 14.0
    height_attribution: float = 2.5
    height_per_bed_track: float = 0.4
    track_spacing: float = 0.1
    positive_color: str = '#E74C3C'
    negative_color: str = '#3498DB'
    y_quantile_lower: float = 0.001
    y_quantile_upper: float = 0.999



def load_bedgraph(bedgraph_path: str) -> Optional[BedGraphData]:
    if not os.path.exists(bedgraph_path):
        print(f"[WARN] Bedgraph not found: {bedgraph_path}")
        return None
    
    df = pd.read_csv(bedgraph_path, sep='\t', header=None,
                     names=['chr', 'start', 'end', 'value', 'gene', 'pert'])
    
    return BedGraphData(
        chromosomes=df['chr'].values,
        starts=df['start'].values.astype(int),
        ends=df['end'].values.astype(int),
        values=df['value'].values.astype(float),
        genes=df['gene'].values.astype(str),
        perts=df['pert'].values.astype(str)
    )


def load_bed_track(bed_path: str, track_name: str, color: str, 
                   label: str, gene_col: Optional[int] = None) -> Optional[BedTrackData]:
    if not os.path.exists(bed_path):
        print(f"[WARN] BED file not found: {bed_path}")
        return None
    
    try:
        df = pd.read_csv(bed_path, sep='\t', header=None, comment='#')
    except Exception as e:
        print(f"[WARN] Error reading {bed_path}: {e}")
        return None
    
    if len(df) == 0:
        print(f"[WARN] Empty BED file: {bed_path}")
        return None
    
    chroms = df.iloc[:, 0].values.astype(str)
    starts = df.iloc[:, 1].values.astype(int)
    ends = df.iloc[:, 2].values.astype(int)
    
    genes = np.array([''] * len(df))
    if gene_col is not None and gene_col < len(df.columns):
        genes = df.iloc[:, gene_col].values.astype(str)
    elif len(df.columns) >= 4:
        for col_idx in [3, -1, -2]:
            try:
                col = df.iloc[:, col_idx].values.astype(str)
                if not all(str(v).replace('.', '').replace('-', '').isdigit() for v in col[:min(10, len(col))]):
                    genes = col
                    break
            except:
                continue
    
    return BedTrackData(
        name=track_name,
        chromosomes=chroms,
        starts=starts,
        ends=ends,
        genes=genes,
        color=color,
        label=label
    )


def load_all_bed_tracks(study: str, pert: str, tf_symbol: str) -> List[BedTrackData]:
    bed_configs = get_bed_track_configs(study, pert, tf_symbol)

    tracks = []
    for cfg in bed_configs:
        track = load_bed_track(
            cfg['path'],
            cfg['name'],
            cfg['color'],
            cfg['label'],
            cfg.get('gene_col')
        )
        if track is not None:
            tracks.append(track)
            print(f"  Loaded {cfg['name']}: {len(track.chromosomes)} regions")

    return tracks


def get_gene_regions_from_bedgraph(bedgraph_data: BedGraphData, gene: str) -> Tuple[str, int, int]:
    mask = bedgraph_data.genes == gene
    if not np.any(mask):
        return None, None, None
    
    chroms = bedgraph_data.chromosomes[mask]
    starts = bedgraph_data.starts[mask]
    ends = bedgraph_data.ends[mask]
    
    return chroms[0], int(starts.min()), int(ends.max())


def filter_track_by_region(track: BedTrackData, chrom: str, 
                           region_start: int, region_end: int,
                           gene: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    mask = track.chromosomes == chrom
    
    mask &= (track.starts < region_end) & (track.ends > region_start)
    
    if gene is not None and track.name != 'chip_seq':
        gene_mask = np.array([gene in str(g) for g in track.genes])
        mask &= gene_mask
    
    return track.starts[mask], track.ends[mask]


def count_regions_for_gene(track: BedTrackData, gene: str,
                           gene_window: Optional[Tuple[str, int, int]] = None) -> int:
    has_gene_annotation = (
        track.genes is not None and 
        len(track.genes) > 0 and 
        any(g != '' for g in track.genes[:min(10, len(track.genes))])
    )
    
    if has_gene_annotation:
        gene_mask = np.array([gene in str(g) for g in track.genes])
        return gene_mask.sum()
    elif gene_window is not None:
        chrom, win_start, win_end = gene_window
        overlap_mask = (
            (track.chromosomes == chrom) &
            (track.starts < win_end) &
            (track.ends > win_start)
        )
        return overlap_mask.sum()
    else:
        return 0



def ensure_dir(filepath: str):
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def plot_attribution_with_bed_comparison(
    bedgraph_data: BedGraphData,
    bed_tracks: List[BedTrackData],
    output_path: str,
    gene: str,
    pert: str,
    config: Optional[PlotConfig] = None,
    title: Optional[str] = None,
    show_shuffle: bool = True,
    gene_window: Optional[Tuple[str, int, int]] = None
):
    ensure_dir(output_path)
    plt.rcParams.update(PLOT_STYLE)
    config = config or PlotConfig()
    
    mask = (bedgraph_data.genes == gene) & (bedgraph_data.perts == pert)
    if not np.any(mask):
        print(f"[WARN] No bedgraph data for gene={gene}, pert={pert}")
        return False
    
    chroms = bedgraph_data.chromosomes[mask]
    starts = bedgraph_data.starts[mask]
    ends = bedgraph_data.ends[mask]
    values = bedgraph_data.values[mask]
    
    sort_idx = np.argsort(starts)
    chroms = chroms[sort_idx]
    starts = starts[sort_idx]
    ends = ends[sort_idx]
    values = values[sort_idx]
    
    chrom = chroms[0]
    region_start = int(starts.min())
    region_end = int(ends.max())
    
    if gene_window is None:
        gene_window = (chrom, region_start, region_end)
    
    if not show_shuffle:
        bed_tracks = [t for t in bed_tracks if 'shuffle' not in t.name]
    
    n_bed_tracks = len(bed_tracks)
    total_height = config.height_attribution + n_bed_tracks * config.height_per_bed_track + 0.5
    
    fig = plt.figure(figsize=(config.width, total_height))
    
    height_ratios = [config.height_attribution] + [config.height_per_bed_track] * n_bed_tracks
    gs = fig.add_gridspec(1 + n_bed_tracks, 1, height_ratios=height_ratios, hspace=0.05)
    
    ax_attr = fig.add_subplot(gs[0])
    
    x_positions = (starts + ends) / 2
    y_positive = np.where(values >= 0, values, 0)
    y_negative = np.where(values < 0, values, 0)
    
    ax_attr.fill_between(x_positions, 0, y_positive, 
                         color=config.positive_color, alpha=0.7, linewidth=0)
    ax_attr.fill_between(x_positions, y_negative, 0, 
                         color=config.negative_color, alpha=0.7, linewidth=0)
    
    all_values = values[~np.isnan(values)]
    if len(all_values) > 0:
        y_lower = np.quantile(all_values, config.y_quantile_lower)
        y_upper = np.quantile(all_values, config.y_quantile_upper)
        y_range = y_upper - y_lower
        y_lower = y_lower - y_range * 0.1
        y_upper = y_upper + y_range * 0.1
        y_lower = min(y_lower, -abs(y_upper) * 0.1)
        y_upper = max(y_upper, abs(y_lower) * 0.1)
        ax_attr.set_ylim(y_lower, y_upper)
    
    ax_attr.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax_attr.set_ylabel('Attribution\n(fold-change)', fontsize=10)
    ax_attr.set_xlim(region_start, region_end)
    ax_attr.tick_params(labelbottom=False)
    
    if title:
        ax_attr.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax_attr.set_title(f'{gene} × {pert}', fontsize=12, fontweight='bold')
    
    for i, track in enumerate(bed_tracks):
        ax_bed = fig.add_subplot(gs[i + 1], sharex=ax_attr)
        
        track_starts, track_ends = filter_track_by_region(
            track, chrom, region_start, region_end, gene
        )
        
        for s, e in zip(track_starts, track_ends):
            rect = mpatches.Rectangle(
                (s, 0.1), e - s, 0.8,
                facecolor=track.color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax_bed.add_patch(rect)
        
        ax_bed.set_ylim(0, 1)
        ax_bed.set_xlim(region_start, region_end)
        ax_bed.set_yticks([])
        
        ax_bed.set_ylabel(track.label, fontsize=8, rotation=0, 
                          ha='right', va='center', labelpad=5)
        
        n_regions = count_regions_for_gene(track, gene, gene_window)
        
        ax_bed.text(region_end, 0.5, f'  n={n_regions}', 
                    fontsize=7, va='center', ha='left')
        
        if i < n_bed_tracks - 1:
            ax_bed.tick_params(labelbottom=False)
        else:
            ax_bed.set_xlabel(f'Genomic Position ({chrom})', fontsize=10)
            ax_bed.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))
        
        ax_bed.spines['top'].set_visible(False)
        ax_bed.spines['right'].set_visible(False)
        ax_bed.spines['left'].set_visible(False)
    
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return True


def plot_bed_comparison_summary(
    bed_tracks: List[BedTrackData],
    genes: List[str],
    output_path: str,
    gene_windows: Optional[Dict[str, Tuple[str, int, int]]] = None,
    title: Optional[str] = None,
    show_shuffle: bool = False
):
    ensure_dir(output_path)
    plt.rcParams.update(PLOT_STYLE)
    
    if not show_shuffle:
        bed_tracks = [t for t in bed_tracks if 'shuffle' not in t.name]
    
    data = []
    for track in bed_tracks:
        counts = []
        for gene in genes:
            gene_window = gene_windows.get(gene) if gene_windows else None
            count = count_regions_for_gene(track, gene, gene_window)
            counts.append(count)
        data.append(counts)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(max(12, len(genes) * 0.5), len(bed_tracks) * 0.6 + 2))
    
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(genes)))
    ax.set_yticks(np.arange(len(bed_tracks)))
    ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([t.label for t in bed_tracks], fontsize=9)
    
    for i in range(len(bed_tracks)):
        for j in range(len(genes)):
            text = ax.text(j, i, str(data[i, j]),
                          ha='center', va='center', fontsize=7,
                          color='white' if data[i, j] > data.max() / 2 else 'black')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of regions', fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title('BED Region Counts per Gene', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary: {output_path}")



def process_single_perturbation(
    pert: str,
    study: str,
    study_name: str,
    n_genes: int = 20,
    show_shuffle: bool = True,
    df3: Optional[pd.DataFrame] = None
) -> Tuple[int, int]:
    tf_symbol = pert.split(".")[-1]
    
    bedgraph_path = f"attribution/{study}/{pert}/{pert}_attribution_fc.bedgraph"
    
    output_base = f"track_comparison/{study}/{pert}"
    
    print("\n" + "=" * 70)
    print(f"Processing: {pert}")
    print("=" * 70)
    print(f"TF symbol: {tf_symbol}")
    print(f"Output: {output_base}")
    
    print("\n  [1] Loading bedgraph data...")
    bedgraph_data = load_bedgraph(bedgraph_path)
    if bedgraph_data is None:
        print(f"  [SKIP] Could not load bedgraph data for {pert}")
        return 0, 0
    print(f"      Loaded {len(bedgraph_data.chromosomes)} bins")
    
    unique_genes = list(set(bedgraph_data.genes))
    print(f"      Found {len(unique_genes)} unique genes")
    
    if df3 is not None and pert in df3.columns:
        top_genes = (
            df3.abs()
               .sort_values(pert, ascending=False)[pert]
               .head(n_genes)
               .index
               .to_list()
        )
        top_genes = [g for g in top_genes if g in unique_genes]
        print(f"      Selected top {len(top_genes)} genes by effect size")
    else:
        top_genes = unique_genes[:n_genes]
        print(f"      Using first {len(top_genes)} genes (no prediction data)")
    
    if len(top_genes) == 0:
        print(f"  [SKIP] No genes found for {pert}")
        return 0, 0
    
    gene_windows = {}
    for gene in top_genes:
        chrom, win_start, win_end = get_gene_regions_from_bedgraph(bedgraph_data, gene)
        if chrom is not None:
            gene_windows[gene] = (chrom, win_start, win_end)
    
    print("\n  [2] Loading BED tracks...")
    bed_tracks = load_all_bed_tracks(study, pert, tf_symbol)
    
    if len(bed_tracks) == 0:
        print(f"  [SKIP] No BED tracks loaded for {pert}")
        return 0, 0
    
    print(f"      Loaded {len(bed_tracks)} BED tracks")
    
    
    os.makedirs(output_base, exist_ok=True)
    
    print("\n  [3] Creating summary plot...")
    summary_path = os.path.join(output_base, f"{pert}_bed_summary.svg")
    plot_bed_comparison_summary(
        bed_tracks=bed_tracks,
        genes=top_genes,
        output_path=summary_path,
        gene_windows=gene_windows,
        title=f"BED Region Summary: {pert}",
        show_shuffle=show_shuffle
    )
    
    print(f"\n  [4] Creating per-gene plots for {len(top_genes)} genes...")
    config = PlotConfig()
    
    successful = 0
    for i, gene in enumerate(top_genes):
        print(f"      [{i+1}/{len(top_genes)}] {gene}")
        
        gene_output_dir = os.path.join(output_base, gene)
        os.makedirs(gene_output_dir, exist_ok=True)
        
        output_path = os.path.join(gene_output_dir, f"{gene}_{pert}_comparison.svg")
        
        gene_window = gene_windows.get(gene)
        
        success = plot_attribution_with_bed_comparison(
            bedgraph_data=bedgraph_data,
            bed_tracks=bed_tracks,
            output_path=output_path,
            gene=gene,
            pert=pert,
            config=config,
            title=f"Attribution & BED Comparison: {gene} × {pert}",
            show_shuffle=show_shuffle,
            gene_window=gene_window
        )
        
        if success:
            successful += 1
    
    gene_list_path = os.path.join(output_base, f"{pert}_plotted_genes.txt")
    with open(gene_list_path, 'w') as f:
        f.write('\n'.join(top_genes))
    
    print(f"\n  [Done] {pert}: {successful}/{len(top_genes)} gene plots")
    
    return successful, len(top_genes)


def main():
    
    
    study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
    pretrained_model = "alphagenome"  # Options: "alphagenome", "borzoi", "enformer"
    study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
    study = f"{study_name}__{study_suffix}"
    
    all_perts = [
        "Norman.AHR",
        "Norman.CEBPA",
        "Norman.CEBPB",
        "Norman.EGR1",
        "Norman.ETS2",
        "Norman.FOXA1",
        "Norman.HNF4A",
        "Norman.IRF1",
        "Norman.JUN",
        "Norman.KMT2A",
        "Norman.PRDM1",
        "Norman.SNAI1",
        "Norman.SPI1",
        "Norman.TP73",
    ]
    
    n_genes = 20
    
    show_shuffle = True
    
    
    print("=" * 70)
    print("Attribution Track with BED Comparison - Batch Processing")
    print("=" * 70)
    print(f"Study: {study}")
    print(f"Perturbations to process: {len(all_perts)}")
    print(f"Genes per perturbation: {n_genes}")
    print(f"Show shuffle controls: {show_shuffle}")
    print("=" * 70)
    
    
    pred_path = f"data/{study_name}_enformer.tsv"
    pred_npy_path = f"prediction/{study}/prediction.npy"
    
    df3 = None
    if os.path.exists(pred_path) and os.path.exists(pred_npy_path):
        print("\n[Pre-load] Loading prediction data...")
        df = pd.read_csv(pred_path, sep="\t", index_col=0)
        pred = np.load(pred_npy_path)
        
        df2 = pd.DataFrame(pred, index=df.index)
        df2.columns = df.columns[1:]
        

        df_val = df[df.columns[1:]]
        ctrl = df_val.columns[0]
        df3 = (df_val.T - df_val[ctrl]).T.drop(ctrl, axis=1)

        print(f"  Loaded predictions for {len(df3.columns)} perturbations")
    else:
        print("\n[Pre-load] Prediction data not found, will use default gene ordering")
    
    
    results = {}
    
    for pert_idx, pert in enumerate(all_perts):
        print(f"\n\n{'#' * 70}")
        print(f"# [{pert_idx + 1}/{len(all_perts)}] {pert}")
        print(f"{'#' * 70}")
        
        try:
            successful, total = process_single_perturbation(
                pert=pert,
                study=study,
                study_name=study_name,
                n_genes=n_genes,
                show_shuffle=show_shuffle,
                df3=df3
            )
            results[pert] = {'successful': successful, 'total': total, 'status': 'OK'}
        except Exception as e:
            print(f"  [ERROR] Failed to process {pert}: {e}")
            results[pert] = {'successful': 0, 'total': 0, 'status': f'ERROR: {e}'}
    
    
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_successful = 0
    total_genes = 0
    
    print(f"\n{'Perturbation':<35} {'Status':<10} {'Plots':<15}")
    print("-" * 60)
    
    for pert in all_perts:
        r = results.get(pert, {'successful': 0, 'total': 0, 'status': 'NOT RUN'})
        status = 'OK' if r['status'] == 'OK' else 'SKIP/ERR'
        plots = f"{r['successful']}/{r['total']}" if r['total'] > 0 else "N/A"
        print(f"{pert:<35} {status:<10} {plots:<15}")
        
        total_successful += r['successful']
        total_genes += r['total']
    
    print("-" * 60)
    print(f"{'TOTAL':<35} {'':<10} {total_successful}/{total_genes}")
    print("=" * 70)
    
    summary_path = f"track_comparison/{study}/processing_summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write("Perturbation\tStatus\tSuccessful\tTotal\n")
        for pert in all_perts:
            r = results.get(pert, {'successful': 0, 'total': 0, 'status': 'NOT RUN'})
            f.write(f"{pert}\t{r['status']}\t{r['successful']}\t{r['total']}\n")
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()