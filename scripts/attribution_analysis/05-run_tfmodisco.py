#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

# TF-MoDISco import (modisco-lite v2 API)
try:
    import modiscolite
    from modiscolite import tfmodisco, io as modisco_io
    HAS_MODISCO = True
except ImportError:
    HAS_MODISCO = False

from genperturb.dataloaders._genome import GenomeIntervalDataset



def run_tfmodisco_workflow(condition, hypothetical_contribs, onehot_data):
    """
    Run TF-MoDISco using modisco-lite (v2) API.

    Parameters
    ----------
    condition : str
        Parameter preset: "default", "short", or "long"
    hypothetical_contribs : np.ndarray
        Hypothetical contribution scores (saliency/gradient), shape (n, seq_len, 4).
        modisco-lite computes actual contribs internally as one_hot * hypothetical_contribs.
    onehot_data : np.ndarray
        One-hot encoded sequences, shape (n, seq_len, 4)
    """
    if condition == "default":
        sliding_window_size = 21
        flank_size = 10
        target_seqlet_fdr = 0.3
        trim_to_window_size = 30
        initial_flank_to_add = 10
        final_flank_to_add = 0
        final_min_cluster_size = 100
    elif condition == "short":
        sliding_window_size = 15
        flank_size = 5
        target_seqlet_fdr = 0.2
        trim_to_window_size = 15
        initial_flank_to_add = 5
        final_flank_to_add = 5
        final_min_cluster_size = 20
    elif condition == "long":
        sliding_window_size = 15
        flank_size = 5
        target_seqlet_fdr = 0.2
        trim_to_window_size = 6
        initial_flank_to_add = 10
        final_flank_to_add = 0
        final_min_cluster_size = 20
    else:
        raise ValueError(f"Unknown condition: {condition}")

    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        one_hot=onehot_data,
        hypothetical_contribs=hypothetical_contribs,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        target_seqlet_fdr=target_seqlet_fdr,
        trim_to_window_size=trim_to_window_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        final_min_cluster_size=final_min_cluster_size,
        subcluster_perplexity=10,
        verbose=True,
    )
    return pos_patterns, neg_patterns


def process_html_to_dataframe(filename, report_dir, pert):
    """Parse modiscolite v2 report HTML and extract TOMTOM match table.

    v2 report structure:
      Table 0: summary (Pattern, Seqlets, ..., Match 0, Q-value, ...)
      Then per pattern: [Metric table, Rank/Match/Logo/Q-value table] pairs
    The detail Rank tables contain all N matches (up to -n), while the
    summary table only shows top 3. We use the detail tables.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("[WARN] beautifulsoup4 not installed, skipping HTML parsing")
        return

    with open(filename, "r") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        print(f"[WARN] No tables found in {filename}")
        return

    # Extract pattern names and seqlet counts from summary table (table 0)
    summary_table = tables[0]
    patterns_info = []
    for tr in summary_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        pattern_span = tds[0].find("span", class_="pattern-id")
        pattern_name = pattern_span.get_text().strip() if pattern_span else tds[0].get_text().strip()
        num_seqlets = tds[1].get_text().strip()
        patterns_info.append((pattern_name, num_seqlets))

    # Collect detail Rank/Match tables (headers: Rank, Match, Logo, Q-value)
    rank_tables = []
    for table in tables[1:]:
        headers = [th.get_text().strip() for th in table.find_all("th")]
        if "Rank" in headers and "Match" in headers:
            rank_tables.append(table)

    reshaped_rows = []
    for idx, (pattern_name, num_seqlets) in enumerate(patterns_info):
        if idx < len(rank_tables):
            for tr in rank_tables[idx].find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue
                match_val = tds[1].get_text().strip()
                qval_val = tds[3].get_text().strip()
                if match_val and qval_val:
                    reshaped_rows.append({
                        "pattern": pattern_name,
                        "num_seqlets": num_seqlets,
                        "match": match_val,
                        "qval": qval_val,
                    })

    reshaped_df = pd.DataFrame(
        reshaped_rows, columns=["pattern", "num_seqlets", "match", "qval"]
    )
    reshaped_df["qval"] = pd.to_numeric(reshaped_df["qval"], errors="coerce")
    reshaped_df = reshaped_df.dropna(subset=["qval"])
    reshaped_df["qval"] = reshaped_df["qval"].astype("float32")
    reshaped_df["perturbation"] = pert

    out_path = f"{report_dir}/{pert}_MA_list.txt"
    reshaped_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Motif list saved: {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Run TF-MoDISco on captum+peak pipeline output"
    )
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--study-suffix", required=True, help="Study suffix")
    parser.add_argument("--pert", required=True, help="Perturbation name")
    parser.add_argument(
        "--input-base", default="attribution_analysis",
        help="Base directory for input files (default: attribution_analysis)"
    )
    parser.add_argument(
        "--output-base", default="attribution_analysis",
        help="Base directory for output files (default: attribution_analysis)"
    )
    parser.add_argument(
        "--condition", default="short",
        choices=["default", "short", "long"],
        help="TF-MoDISco parameter condition (default: short)"
    )
    parser.add_argument(
        "--attribution", default="ixg",
        help="Attribution method key in H5 (default: ixg)"
    )
    parser.add_argument(
        "--context-length", type=int, default=128,
        help="Window size for one-hot encoding (default: 128)"
    )
    parser.add_argument(
        "--fasta", default="fasta/GRCh38.p14.genome.fa",
        help="Genome FASTA file"
    )
    parser.add_argument(
        "--meme-motif",
        default="reference/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme_modified.txt",
        help="MEME-format motif file for modisco report"
    )
    args = parser.parse_args()

    if not HAS_MODISCO:
        print("ERROR: modisco-lite not installed. Install with: pip install modisco-lite")
        sys.exit(1)

    study_full = f"{args.study}__{args.study_suffix}"
    safe_pert = args.pert.replace("/", "_")

    # Locate input files
    input_dir = Path(args.input_base) / "attribution" / study_full / safe_pert
    attr_h5 = input_dir / f"{safe_pert}.h5"
    peaks_bed = input_dir / f"{safe_pert}_peaks_bin128bp.bed"

    for f, label in [(attr_h5, "Attribution H5"), (peaks_bed, "Peaks BED")]:
        if not f.exists():
            print(f"[ERROR] {label} not found: {f}")
            sys.exit(1)

    if not Path(args.fasta).exists():
        print(f"[ERROR] FASTA not found: {args.fasta}")
        sys.exit(1)

    # Output directory
    modisco_dir = Path(args.output_base) / "tfmodisco" / study_full / safe_pert
    modisco_dir.mkdir(parents=True, exist_ok=True)

    modisco_h5 = modisco_dir / f"{safe_pert}_modisco.h5"
    report_dir = modisco_dir / "modisco_result"

    # Clean stale log files
    for log_name in ["no_motif_meme.log", "no_motif_modisco.log"]:
        log_path = modisco_dir / log_name
        if log_path.exists():
            log_path.unlink()

    print("=" * 60)
    print(f"TF-MoDISco: {safe_pert}")
    print("=" * 60)
    print(f"  Study:       {study_full}")
    print(f"  Attribution: {attr_h5}")
    print(f"  Peaks BED:   {peaks_bed}")
    print(f"  Condition:   {args.condition}")
    print(f"  Output:      {modisco_dir}")
    print("=" * 60)

    print("\n--- Loading one-hot sequences ---")
    ds = GenomeIntervalDataset(
        bed_file=str(peaks_bed),
        fasta_file=args.fasta,
        context_length=args.context_length,
    )
    onehot_data = np.array([ds[i].detach().numpy().astype("float32") for i in range(len(ds))])
    print(f"  Loaded {onehot_data.shape[0]} sequences of length {args.context_length}")

    print("\n--- Loading attribution scores ---")
    sal_key = "saliency"

    with h5py.File(attr_h5, "r") as f:
        if sal_key not in f:
            print(f"[ERROR] Key '{sal_key}' not found in {attr_h5}")
            print(f"  Available keys: {list(f.keys())}")
            sys.exit(1)

        hypothetical_contribs = np.array(f[sal_key])  # (n, seq_len, 4)
        print(f"  Loaded {hypothetical_contribs.shape[0]} attribution windows")

    # Verify alignment
    if onehot_data.shape[0] != hypothetical_contribs.shape[0]:
        print(f"[WARN] Mismatch: {onehot_data.shape[0]} sequences vs "
              f"{hypothetical_contribs.shape[0]} attribution windows")
        n = min(onehot_data.shape[0], hypothetical_contribs.shape[0])
        onehot_data = onehot_data[:n]
        hypothetical_contribs = hypothetical_contribs[:n]
        print(f"  Truncated to {n} entries")

    print("\n--- Running TF-MoDISco workflow ---")
    try:
        pos_patterns, neg_patterns = run_tfmodisco_workflow(
            args.condition, hypothetical_contribs, onehot_data
        )
        modisco_io.save_hdf5(
            str(modisco_h5), pos_patterns, neg_patterns,
            window_size=args.context_length,
        )
        print(f"  Saved: {modisco_h5}")
    except Exception as e:
        print(f"[ERROR] TF-MoDISco failed: {e}")
        with open(modisco_dir / "no_motif_modisco.log", "w") as f:
            f.write(f"An error occurred: {e}\n")
        # Continue to exit gracefully (no motifs is not a fatal error)
        print("  Continuing without motif results...")
        return

    print("\n--- Generating report ---")
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_cmd = (
            f"modisco report -i {modisco_h5} -o {report_dir} "
            f"-m {args.meme_motif} -n 100"
        )
        print(f"  {report_cmd}")
        subprocess.check_output(report_cmd, shell=True, stderr=subprocess.STDOUT)

        # Parse HTML report
        html_candidates = [
            report_dir / "report.html",
            report_dir / "motifs.html",
            report_dir / "index.html",
        ]
        html_file = next((c for c in html_candidates if c.exists()), None)
        if html_file is not None:
            process_html_to_dataframe(str(html_file), str(report_dir), safe_pert)
        else:
            print(f"[WARN] No report HTML found in {report_dir}")
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        with open(modisco_dir / "no_motif_meme.log", "w") as f:
            f.write(f"An error occurred: {e}\n")

    print(f"\n{'='*60}")
    print(f"Output: {modisco_dir}")
    if modisco_dir.exists():
        for p in sorted(modisco_dir.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                print(f"  {p.relative_to(modisco_dir)} ({size:,} bytes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
