#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import argparse
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import h5py



def _sum_over_bases(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.sum(axis=1)
    return x


def compute_binned_attribution_np(
    attribution_fc: np.ndarray,
    bin_size: int = 128,
    abs_before_mean: bool = False,
) -> np.ndarray:
    x = _sum_over_bases(attribution_fc).astype(np.float32, copy=False)
    if abs_before_mean:
        x = np.abs(x)

    L = (x.shape[0] // bin_size) * bin_size
    if L <= 0:
        return np.array([], dtype=np.float32)

    return x[:L].reshape(-1, bin_size).mean(axis=1).astype(np.float32, copy=False)



def bedgraph_base(
    chromosome: str,
    seq_start: int,
    n_bins: int,
    bin_size: int,
) -> pd.DataFrame:
    starts = seq_start + np.arange(n_bins, dtype=np.int64) * int(bin_size)
    ends = starts + int(bin_size)
    return pd.DataFrame({"chr": chromosome, "start": starts, "end": ends})


def bedgraph_with_values(base_df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = base_df.copy()
    out["value"] = values.astype(np.float32, copy=False)
    return out


def bedgraph_for_peakcall(
    base_df: pd.DataFrame,
    values: np.ndarray,
    seq_start: int,
    chromosome: str,
    fill_gaps: bool = True,
    abs_after_mean: bool = True,
) -> pd.DataFrame:
    v = np.abs(values) if abs_after_mean else values
    df = bedgraph_with_values(base_df, v)

    if fill_gaps and seq_start > 0:
        gap = pd.DataFrame(
            {"chr": [chromosome], "start": [0], "end": [seq_start], "value": [0.0]}
        )
        df = pd.concat([gap, df], ignore_index=True)

    return df



def run_macs3_peakcall(
    bedgraph_df: pd.DataFrame,
    tmp_dir: Path,
    cutoff: float = 0.0001,
    min_length: int = 100,
    max_gap: int = 150,
) -> pd.DataFrame:
    empty_cols = ["chr", "start", "end", "name", "score", "strand", "signal", "pvalue", "qvalue", "peak"]

    if bedgraph_df.empty:
        return pd.DataFrame(columns=empty_cols)

    vmax = float(np.max(np.abs(bedgraph_df["value"].to_numpy(dtype=np.float32, copy=False)))) if len(bedgraph_df) else 0.0
    if vmax == 0.0:
        return pd.DataFrame(columns=empty_cols)

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    temp_bedgraph = tmp_dir / "input.bedgraph"
    output_file = tmp_dir / "peaks.narrowPeak"

    try:
        bedgraph_df.to_csv(temp_bedgraph, sep="\t", header=False, index=False)

        cmd = [
            "macs3", "bdgpeakcall",
            "-i", str(temp_bedgraph),
            "-o", str(output_file),
            "-c", str(cutoff),
            "-l", str(min_length),
            "-g", str(max_gap),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if output_file.exists() and output_file.stat().st_size > 0:
            peaks_df = pd.read_csv(output_file, sep="\t", header=None, names=empty_cols, comment="#")
            peaks_df = peaks_df[~peaks_df["chr"].astype(str).str.startswith("track")]
            peaks_df = peaks_df.dropna(subset=["chr", "start", "end"])
            if not peaks_df.empty:
                peaks_df["start"] = peaks_df["start"].astype(int)
                peaks_df["end"] = peaks_df["end"].astype(int)
            return peaks_df

        return pd.DataFrame(columns=empty_cols)

    except FileNotFoundError as e:
        raise RuntimeError(
            "MACS3 (macs3) not found."
            "This script requires MACS3."
            "Example: pip install macs3 or install MACS3 in your environment and ensure it is added to your PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr.strip() if e.stderr else "MACS3 error / no peaks")
        print(f"  MACS3 warning: {msg}")
        return pd.DataFrame(columns=empty_cols)
    finally:
        for p in (temp_bedgraph, output_file):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass



@dataclass
class WindowAttribution:
    attribution: np.ndarray          # (128, 4)
    attribution_fc: np.ndarray       # (128, 4)
    saliency: np.ndarray             # (128, 4)
    peak_id: str
    chromosome: str
    window_start: int
    window_end: int
    gene: str
    pert: str
    window_idx: int
    additional_attribution: Optional[np.ndarray] = None
    additional_attribution_fc: Optional[np.ndarray] = None
    additional_saliency: Optional[np.ndarray] = None


def _pad_to_window(x: np.ndarray, window_size: int) -> np.ndarray:
    if x.shape[0] == window_size:
        return x
    pad_len = window_size - x.shape[0]
    return np.pad(x, ((0, pad_len), (0, 0)), mode="constant")


def extract_peak_windows(
    attribution: np.ndarray,
    attribution_fc: np.ndarray,
    saliency: np.ndarray,
    peaks_df: pd.DataFrame,
    chromosome: str,
    seq_start: int,
    gene: str,
    pert: str,
    window_size: int = 128,
    additional_attribution: Optional[np.ndarray] = None,
    additional_attribution_fc: Optional[np.ndarray] = None,
    additional_saliency: Optional[np.ndarray] = None,
) -> List[WindowAttribution]:
    windows: List[WindowAttribution] = []
    if peaks_df is None or peaks_df.empty:
        return windows

    L = int(attribution.shape[0])
    has_additional = additional_attribution is not None

    for peak_idx, peak in enumerate(peaks_df.itertuples(index=False), start=0):
        try:
            peak_chr = str(getattr(peak, "chr"))
            peak_start = int(getattr(peak, "start"))
            peak_end = int(getattr(peak, "end"))
        except Exception:
            continue

        if peak_chr != chromosome:
            continue

        rel_start = peak_start - int(seq_start)
        rel_end = peak_end - int(seq_start)

        if rel_start < 0:
            rel_start = 0
        if rel_end > L:
            rel_end = L
        if rel_start >= rel_end:
            continue

        peak_name = getattr(peak, "name", f"peak_{peak_idx}")
        peak_id = f"{gene}_{pert}_{peak_name}"

        window_idx = 0
        for win_start in range(rel_start, rel_end, window_size):
            win_end = min(win_start + window_size, rel_end)

            attr_w = attribution[win_start:win_end]
            attr_fc_w = attribution_fc[win_start:win_end]
            sal_w = saliency[win_start:win_end]

            if (win_end - win_start) < window_size:
                attr_w = _pad_to_window(attr_w, window_size)
                attr_fc_w = _pad_to_window(attr_fc_w, window_size)
                sal_w = _pad_to_window(sal_w, window_size)

            genomic_start = int(seq_start) + int(win_start)
            genomic_end = genomic_start + int(window_size)

            w = WindowAttribution(
                attribution=attr_w,
                attribution_fc=attr_fc_w,
                saliency=sal_w,
                peak_id=peak_id,
                chromosome=chromosome,
                window_start=genomic_start,
                window_end=genomic_end,
                gene=gene,
                pert=pert,
                window_idx=window_idx,
            )

            if has_additional:
                if (
                    additional_attribution is not None
                    and additional_attribution.ndim == 2
                    and additional_attribution.shape[0] == L
                ):
                    add_w = additional_attribution[win_start:win_end]
                    add_fc_w = additional_attribution_fc[win_start:win_end]
                    add_sal_w = additional_saliency[win_start:win_end]
                    if (win_end - win_start) < window_size:
                        add_w = _pad_to_window(add_w, window_size)
                        add_fc_w = _pad_to_window(add_fc_w, window_size)
                        add_sal_w = _pad_to_window(add_sal_w, window_size)
                else:
                    add_w = additional_attribution
                    add_fc_w = additional_attribution_fc
                    add_sal_w = additional_saliency

                w.additional_attribution = add_w
                w.additional_attribution_fc = add_fc_w
                w.additional_saliency = add_sal_w

            windows.append(w)
            window_idx += 1

    return windows



def save_peak_attribution_hdf5(
    filepath: Path,
    window_attributions: List[WindowAttribution],
    attribution_method: str = "ixg",
    compression: str = "gzip",
    compression_opts: int = 4,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not window_attributions:
        print("  Warning: No window attributions to save")
        return

    n = len(window_attributions)
    window_size = int(window_attributions[0].attribution.shape[0])
    n_channels = int(window_attributions[0].attribution.shape[1])

    has_additional = window_attributions[0].additional_attribution is not None
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(filepath, "w") as hf:
        ds_attr = hf.create_dataset(
            f"{attribution_method}",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )
        ds_attr_fc = hf.create_dataset(
            f"{attribution_method}_fc",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )
        ds_attr_abs = hf.create_dataset(
            f"{attribution_method}_abs",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )
        ds_attr_fc_abs = hf.create_dataset(
            f"{attribution_method}_fc_abs",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )
        ds_sal = hf.create_dataset(
            "saliency",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )
        ds_sal_abs = hf.create_dataset(
            "saliency_abs",
            shape=(n, window_size, n_channels),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
            chunks=True,
        )

        if has_additional:
            add_grp = hf.create_group("additional_input")
            add_ds_attr = add_grp.create_dataset(
                f"{attribution_method}",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )
            add_ds_attr_fc = add_grp.create_dataset(
                f"{attribution_method}_fc",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )
            add_ds_attr_abs = add_grp.create_dataset(
                f"{attribution_method}_abs",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )
            add_ds_attr_fc_abs = add_grp.create_dataset(
                f"{attribution_method}_fc_abs",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )
            add_ds_sal = add_grp.create_dataset(
                "saliency",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )
            add_ds_sal_abs = add_grp.create_dataset(
                "saliency_abs",
                shape=(n, window_size, n_channels),
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                chunks=True,
            )

        meta = hf.create_group("metadata")
        meta_peak_id = meta.create_dataset("peak_id", shape=(n,), dtype=str_dt)
        meta_chr = meta.create_dataset("chromosome", shape=(n,), dtype=str_dt)
        meta_gene = meta.create_dataset("gene", shape=(n,), dtype=str_dt)
        meta_pert = meta.create_dataset("pert", shape=(n,), dtype=str_dt)
        meta_ws = meta.create_dataset("window_start", shape=(n,), dtype=np.int64)
        meta_we = meta.create_dataset("window_end", shape=(n,), dtype=np.int64)
        meta_wi = meta.create_dataset("window_idx", shape=(n,), dtype=np.int32)

        all_attr = np.stack([w.attribution.astype(np.float32, copy=False) for w in window_attributions])
        all_attr_fc = np.stack([w.attribution_fc.astype(np.float32, copy=False) for w in window_attributions])
        all_sal = np.stack([w.saliency.astype(np.float32, copy=False) for w in window_attributions])

        ds_attr[:] = all_attr
        ds_attr_fc[:] = all_attr_fc
        ds_attr_abs[:] = np.abs(all_attr)
        ds_attr_fc_abs[:] = np.abs(all_attr_fc)
        ds_sal[:] = all_sal
        ds_sal_abs[:] = np.abs(all_sal)

        del all_attr, all_attr_fc, all_sal

        if has_additional:
            all_add_attr = np.stack([w.additional_attribution.astype(np.float32, copy=False) for w in window_attributions])
            all_add_attr_fc = np.stack([w.additional_attribution_fc.astype(np.float32, copy=False) for w in window_attributions])
            all_add_sal = np.stack([w.additional_saliency.astype(np.float32, copy=False) for w in window_attributions])

            add_ds_attr[:] = all_add_attr
            add_ds_attr_fc[:] = all_add_attr_fc
            add_ds_attr_abs[:] = np.abs(all_add_attr)
            add_ds_attr_fc_abs[:] = np.abs(all_add_attr_fc)
            add_ds_sal[:] = all_add_sal
            add_ds_sal_abs[:] = np.abs(all_add_sal)

            del all_add_attr, all_add_attr_fc, all_add_sal

        meta_peak_id[:] = [w.peak_id for w in window_attributions]
        meta_chr[:] = [w.chromosome for w in window_attributions]
        meta_gene[:] = [w.gene for w in window_attributions]
        meta_pert[:] = [w.pert for w in window_attributions]
        meta_ws[:] = np.array([int(w.window_start) for w in window_attributions], dtype=np.int64)
        meta_we[:] = np.array([int(w.window_end) for w in window_attributions], dtype=np.int64)
        meta_wi[:] = np.array([int(w.window_idx) for w in window_attributions], dtype=np.int32)

        hf.attrs["window_size"] = int(window_size)
        hf.attrs["n_windows"] = int(n)
        hf.attrs["has_additional_input"] = bool(has_additional)


def save_bed_file(df: pd.DataFrame, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, sep="\t", header=False, index=False)


def save_window_bed_file(window_attributions: List[WindowAttribution], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not window_attributions:
        print("  Warning: No windows to save for BED file")
        return

    rows = []
    for w in window_attributions:
        rows.append([w.chromosome, w.window_start, w.window_end, f"{w.peak_id}_w{w.window_idx}", ".", "+"])

    df = pd.DataFrame(rows, columns=["chr", "start", "end", "name", "score", "strand"])
    df.to_csv(filepath, sep="\t", header=False, index=False)



def process_raw_attribution_h5(
    h5_path: Path,
    macs_cutoff: float = 0.0001,
    bin_size: int = 128,
):
    with h5py.File(h5_path, "r") as hf:
        mode = hf.attrs.get("mode", "unknown")
        attribution_method = hf.attrs.get("attribution_method", "ixg")

        if mode == "across_perturbations":
            file_gene = hf.attrs.get("gene", "")
            file_chromosome = hf.attrs.get("chromosome", "")
            file_seq_start = int(hf.attrs.get("seq_start", 0))
            file_seq_end = int(hf.attrs.get("seq_end", 0))
            pert_from_file = None
        else:
            pert_from_file = hf.attrs.get("pert", "")
            file_gene = None
            file_chromosome = None
            file_seq_start = None
            file_seq_end = None

        group_names = [k for k in hf.keys() if isinstance(hf[k], h5py.Group)]

        all_windows: List[WindowAttribution] = []
        all_peaks: List[pd.DataFrame] = []
        all_bedgraph: List[pd.DataFrame] = []
        all_bedgraph_abs: List[pd.DataFrame] = []

        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_root = Path(tmp_root)

            for group_name in group_names:
                grp = hf[group_name]

                attribution = grp[attribution_method][:]
                attribution_fc = grp[f"{attribution_method}_fc"][:]
                saliency = grp["saliency"][:]

                if mode == "across_perturbations":
                    gene = file_gene
                    pert = grp.attrs.get("pert", group_name)
                    chromosome = file_chromosome
                    seq_start = int(file_seq_start)
                    seq_end = int(file_seq_end)
                else:
                    gene = group_name
                    pert = pert_from_file
                    chromosome = grp.attrs.get("chromosome", "")
                    seq_start = int(grp.attrs.get("seq_start", 0))
                    seq_end = int(grp.attrs.get("seq_end", 0))

                has_additional = "additional_input" in grp
                if has_additional:
                    add_grp = grp["additional_input"]
                    add_attr = add_grp[attribution_method][:]
                    add_attr_fc = add_grp[f"{attribution_method}_fc"][:]
                    add_sal = add_grp["saliency"][:]
                else:
                    add_attr = None
                    add_attr_fc = None
                    add_sal = None

                attr_binned = compute_binned_attribution_np(attribution_fc, bin_size=bin_size, abs_before_mean=False)
                attr_binned_abs_before = compute_binned_attribution_np(attribution_fc, bin_size=bin_size, abs_before_mean=True)

                n_bins = int(attr_binned.shape[0])
                if n_bins == 0:
                    print(f"    {gene} x {pert}: empty binned attribution -> skip")
                    continue

                base = bedgraph_base(chromosome, seq_start, n_bins, bin_size)

                bed_fc = bedgraph_with_values(base, attr_binned)
                bed_fc["gene"] = gene
                bed_fc["pert"] = pert
                all_bedgraph.append(bed_fc)

                bed_fc_abs = bedgraph_with_values(base, attr_binned_abs_before)
                bed_fc_abs["gene"] = gene
                bed_fc_abs["pert"] = pert
                all_bedgraph_abs.append(bed_fc_abs)

                bed_for_call = bedgraph_for_peakcall(
                    base_df=base,
                    values=attr_binned,
                    seq_start=seq_start,
                    chromosome=chromosome,
                    fill_gaps=True,
                    abs_after_mean=True,
                )

                grp_tmp = tmp_root / f"{group_name}".replace("/", "_")
                grp_tmp.mkdir(parents=True, exist_ok=True)

                peaks_df = run_macs3_peakcall(
                    bed_for_call,
                    tmp_dir=grp_tmp,
                    cutoff=macs_cutoff,
                )

                n_peaks = 0 if peaks_df.empty else len(peaks_df)

                windows = extract_peak_windows(
                    attribution=attribution,
                    attribution_fc=attribution_fc,
                    saliency=saliency,
                    peaks_df=peaks_df,
                    chromosome=chromosome,
                    seq_start=seq_start,
                    gene=gene,
                    pert=pert,
                    window_size=128,
                    additional_attribution=add_attr,
                    additional_attribution_fc=add_attr_fc,
                    additional_saliency=add_sal,
                )

                print(f"    {gene} x {pert}: {n_peaks} peaks -> {len(windows)} windows")
                all_windows.extend(windows)

                if not peaks_df.empty:
                    peaks = peaks_df.copy()
                    peaks["gene"] = gene
                    peaks["pert"] = pert
                    all_peaks.append(peaks)

    return all_windows, all_peaks, all_bedgraph, all_bedgraph_abs, attribution_method, mode



def main():
    parser = argparse.ArgumentParser(description="Peak calling on raw attribution HDF5 files (MACS3 required)")
    parser.add_argument("study_name", help="Study name")
    parser.add_argument("study_suffix", help="Study suffix (determines model type)")
    parser.add_argument(
        "mode",
        choices=["variable_genes", "across_genes", "across_perturbations"],
        help="Analysis mode",
    )
    parser.add_argument("--macs_cutoff", type=float, default=0.0001,
                        help="MACS3 cutoff threshold (default: 0.0001)")
    parser.add_argument("--bin_size", type=int, default=128,
                        help="Bin size for attribution aggregation (default: 128)")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory for attribution input/output (default: .)")
    parser.add_argument("--h5-suffix", default="",
                        help="Restrict to *{suffix}_raw_attribution.h5 (e.g. '_gtgenes')")

    args = parser.parse_args()
    study = f"{args.study_name}__{args.study_suffix}"

    base_prefix = Path(args.base_dir)
    if args.mode == "variable_genes":
        base_dir = base_prefix / "attribution" / study
    elif args.mode == "across_genes":
        base_dir = base_prefix / "attribution_pert" / study
    else:
        base_dir = base_prefix / "attribution_seq" / study

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        print("Run 10_captum.py first to generate raw attribution files.")
        sys.exit(1)

    suffix = args.h5_suffix or ""
    h5_files = sorted(base_dir.glob(f"**/*{suffix}_raw_attribution.h5"))
    if suffix:
        h5_files = [p for p in h5_files if p.name.endswith(f"{suffix}_raw_attribution.h5")]
    if not h5_files:
        print(f"Error: No *{suffix}_raw_attribution.h5 files found in {base_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Peak Calling on Raw Attribution")
    print("=" * 70)
    print(f"Study: {study}")
    print(f"Mode: {args.mode}")
    print(f"MACS cutoff: {args.macs_cutoff}")
    print(f"Bin size: {args.bin_size}")
    print(f"Found {len(h5_files)} raw attribution files")
    print("=" * 70)

    for h5_path in h5_files:
        output_dir = h5_path.parent
        print(f"\nProcessing: {h5_path}")

        try:
            all_windows, all_peaks, all_bedgraph, all_bedgraph_abs, attribution_method, mode = process_raw_attribution_h5(
                h5_path,
                macs_cutoff=args.macs_cutoff,
                bin_size=args.bin_size,
            )
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(2)

        stem = h5_path.stem.replace("_raw_attribution", "")

        if all_bedgraph:
            bedgraph_df = pd.concat(all_bedgraph, ignore_index=True)
            save_bed_file(
                bedgraph_df[["chr", "start", "end", "value", "gene", "pert"]],
                output_dir / f"{stem}_attribution_fc.bedgraph",
            )
            print("  Saved attribution bedgraph")

        if all_bedgraph_abs:
            bedgraph_abs_df = pd.concat(all_bedgraph_abs, ignore_index=True)
            save_bed_file(
                bedgraph_abs_df[["chr", "start", "end", "value", "gene", "pert"]],
                output_dir / f"{stem}_attribution_fc_abs.bedgraph",
            )
            print("  Saved absolute attribution bedgraph (abs-before-mean)")

        if all_peaks:
            peaks_df = pd.concat(all_peaks, ignore_index=True)
            save_bed_file(peaks_df, output_dir / f"{stem}_peaks.bed")
            print(f"  Total peaks: {len(peaks_df)}")
        else:
            print("  Warning: No peaks detected")

        if all_windows:
            if args.mode == "variable_genes":
                h5_out = output_dir / f"{stem}.h5"
            else:
                h5_out = output_dir / f"{stem}_attributions.h5"

            save_peak_attribution_hdf5(h5_out, all_windows, attribution_method)
            print(f"  Saved {len(all_windows)} peak-based windows (128bp each) to HDF5: {h5_out.name}")

            if args.mode == "variable_genes":
                bed_filename = f"{stem}_peaks_bin128bp.bed"
                save_window_bed_file(all_windows, output_dir / bed_filename)
                print(f"  Saved window BED file: {bed_filename}")
        else:
            print("  Warning: No peak-based attribution data to save")

        print(f"  Output: {output_dir}")

    print("\n" + "=" * 70)
    print("Peak calling completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
