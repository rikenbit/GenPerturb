from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

_LOCUS_RE = re.compile(r"^(\S+):(\d+)-(\d+)$")


def load_atac_table(tsv_path: str | Path):
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    columns = list(df.columns)
    matrix = df.to_numpy(dtype=np.float32)

    chroms = np.empty(df.shape[0], dtype=object)
    starts = np.empty(df.shape[0], dtype=np.int64)
    ends = np.empty(df.shape[0], dtype=np.int64)
    for i, name in enumerate(df.index):
        m = _LOCUS_RE.match(name)
        if not m:
            chroms[i] = ""
            starts[i] = 0
            ends[i] = 0
            continue
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i] = int(m.group(3))

    by_chrom: dict[str, dict] = {}
    chrom_arr = np.array(chroms)
    for chrom in np.unique(chrom_arr):
        if chrom == "":
            continue
        mask = chrom_arr == chrom
        idx = np.flatnonzero(mask)
        s = starts[idx]
        order = np.argsort(s, kind="stable")
        idx = idx[order]
        by_chrom[chrom] = {
            "starts": starts[idx],
            "ends": ends[idx],
            "row_idx": idx.astype(np.int64),
        }
    return columns, matrix, by_chrom


def _peaks_in_window(by_chrom, chrom: str, view_start: int, view_end: int):
    info = by_chrom.get(chrom)
    if info is None:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    s = info["starts"]
    e = info["ends"]
    right = np.searchsorted(s, view_end, side="left")
    if right == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    s_sub = s[:right]
    e_sub = e[:right]
    keep = e_sub > view_start
    if not keep.any():
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    rows = info["row_idx"][:right][keep]
    return s_sub[keep], e_sub[keep], rows


def window_track(matrix: np.ndarray, by_chrom: dict, columns: list[str],
                  chrom: str, view_start: int, view_end: int, col: str,
                  bin_size: int = 128) -> np.ndarray:
    s, e, rows = _peaks_in_window(by_chrom, chrom, view_start, view_end)
    L = view_end - view_start
    sig = np.zeros(L, dtype=np.float32)
    if rows.size:
        col_idx = columns.index(col)
        vals = matrix[rows, col_idx]
        for ps, pe, v in zip(s, e, vals):
            ls = max(int(ps), view_start) - view_start
            le = min(int(pe), view_end) - view_start
            if le > ls:
                sig[ls:le] += v
    n = (L // bin_size) * bin_size
    if n == 0:
        return sig
    binned = sig[:n].reshape(-1, bin_size).mean(axis=1)
    return binned


def window_coords(view_start: int, view_end: int, bin_size: int = 128) -> np.ndarray:
    L = view_end - view_start
    n = (L // bin_size) * bin_size
    starts = view_start + np.arange(0, n, bin_size)
    return starts + bin_size / 2.0


def seqlet_delta(matrix: np.ndarray, by_chrom: dict, columns: list[str],
                 chrom: str, core_start: int, core_end: int,
                 ctrl_col: str, pert_col: str, flank: int = 0) -> dict:
    s, e, rows = _peaks_in_window(by_chrom, chrom,
                                   core_start - flank, core_end + flank)
    if rows.size == 0:
        return {"n_peaks": 0, "max_abs_delta": 0.0,
                "sum_abs_delta": 0.0, "signed_delta": 0.0}
    ci = columns.index(ctrl_col)
    pi = columns.index(pert_col)
    delta = matrix[rows, pi] - matrix[rows, ci]
    abs_delta = np.abs(delta)
    arg = int(np.argmax(abs_delta))
    return {
        "n_peaks": int(rows.size),
        "max_abs_delta": float(abs_delta[arg]),
        "sum_abs_delta": float(abs_delta.sum()),
        "signed_delta": float(delta[arg]),
    }
