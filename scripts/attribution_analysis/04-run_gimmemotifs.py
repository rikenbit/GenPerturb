#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def create_clean_bed(peaks_bed: Path, output_bed: Path, min_width: int = 50):
    """
    Extract chr/start/end from peaks BED (which may have extra columns)
    and write a clean BED3 file suitable for gimme motifs.

    Also filters out regions narrower than min_width bp.
    """
    df = pd.read_csv(peaks_bed, sep="\t", header=None, comment="#")
    # Use first 3 columns: chr, start, end
    bed3 = df.iloc[:, :3].copy()
    bed3.columns = ["chr", "start", "end"]
    bed3["start"] = bed3["start"].astype(int)
    bed3["end"] = bed3["end"].astype(int)

    # Filter narrow regions
    widths = bed3["end"] - bed3["start"]
    bed3 = bed3[widths >= min_width]

    if bed3.empty:
        return None

    bed3.to_csv(output_bed, sep="\t", header=False, index=False)
    return output_bed


def run_gimmemotifs(
    bed_path: Path,
    output_dir: Path,
    genome: str,
    known_db: str,
    threads: int = 16,
):
    """Run gimme motifs on a BED file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gimme", "motifs",
        str(bed_path),
        str(output_dir),
        "-g", genome,
        "-N", str(threads),
        "--known",
        "-p", known_db,
    ]

    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[STDERR] {result.stderr}")
        # gimme motifs may return non-zero for "no motifs found" etc.
        # Check if output was still produced
        if not any(output_dir.iterdir()):
            raise RuntimeError(f"gimme motifs failed: {result.stderr}")
        else:
            print("[WARN] gimme motifs returned non-zero but produced output")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run GimmeMotifs on captum+peak pipeline output"
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
        "--genome", default="fasta/GRCh38.p14.genome",
        help="Genome path for gimme (without .fa extension)"
    )
    parser.add_argument(
        "--known-db", default="reference/gimmemotifs/motif_db/JASPAR2022_vertebrates.pfm",
        help="Known motif database for gimme"
    )
    parser.add_argument("--threads", type=int, default=16, help="Number of threads")
    parser.add_argument(
        "--bed-type", default="attribution",
        help="BED type: attribution (default), re2g, re2g_extended, abc_score, tss_1kbp"
    )
    args = parser.parse_args()

    study_full = f"{args.study}__{args.study_suffix}"
    safe_pert = args.pert.replace("/", "_")

    # Locate input BED based on bed-type
    if args.bed_type == "attribution":
        input_dir = Path(args.input_base) / "attribution" / study_full / safe_pert
        peaks_bed = input_dir / f"{safe_pert}_peaks.bed"
        peaks_bin_bed = input_dir / f"{safe_pert}_peaks_bin128bp.bed"

        if peaks_bed.exists():
            source_bed = peaks_bed
            print(f"[INFO] Using peaks BED: {peaks_bed}")
        elif peaks_bin_bed.exists():
            source_bed = peaks_bin_bed
            print(f"[INFO] Using peaks_bin128bp BED (fallback): {peaks_bin_bed}")
        else:
            print(f"[ERROR] No peaks BED found in {input_dir}")
            print(f"  Expected: {peaks_bed}")
            print(f"  Or:       {peaks_bin_bed}")
            sys.exit(1)
    else:
        cre_bed = (
            Path(args.input_base) / "cre" / study_full / safe_pert
            / f"{args.bed_type}_{safe_pert}.bed"
        )
        if not cre_bed.exists():
            print(f"[ERROR] CRE BED not found: {cre_bed}")
            sys.exit(1)
        source_bed = cre_bed
        print(f"[INFO] Using CRE BED ({args.bed_type}): {cre_bed}")

    # Check genome and motif DB
    genome_fa = Path(f"{args.genome}.fa")
    if not genome_fa.exists():
        print(f"[ERROR] Genome FASTA not found: {genome_fa}")
        sys.exit(1)

    if not Path(args.known_db).exists():
        print(f"[ERROR] Motif database not found: {args.known_db}")
        sys.exit(1)

    # Create clean BED3 for gimme motifs (strip extra columns)
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_bed = Path(tmpdir) / "peaks_clean.bed"
        result = create_clean_bed(source_bed, clean_bed, min_width=50)

        if result is None:
            print(f"[WARN] No peaks with width >= 50bp. Skipping {safe_pert}.")
            sys.exit(0)

        n_peaks = sum(1 for _ in open(clean_bed))
        print(f"[INFO] {n_peaks} peaks for motif analysis")

        # Output directory
        if args.bed_type == "attribution":
            output_dir = Path(args.output_base) / "gimme_results" / study_full / safe_pert
        else:
            output_dir = (
                Path(args.output_base) / "gimme_results" / study_full
                / safe_pert / args.bed_type
            )

        # Run gimme motifs
        print(f"\n{'='*60}")
        print(f"GimmeMotifs: {safe_pert}")
        print(f"{'='*60}")

        rc = run_gimmemotifs(
            bed_path=clean_bed,
            output_dir=output_dir,
            genome=args.genome,
            known_db=args.known_db,
            threads=args.threads,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"Output: {output_dir}")
    if output_dir.exists():
        output_files = list(output_dir.rglob("*"))
        print(f"Files produced: {len(output_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
