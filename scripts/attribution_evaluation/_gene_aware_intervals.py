#!/usr/bin/env python
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd


Interval = Tuple[int, int, float]
IntervalIndex = Dict[Tuple[str, str], List[Interval]]


def build_interval_index_by_gene(
    intervals_df: pd.DataFrame,
    score_col: str = "score",
) -> IntervalIndex:
    index = defaultdict(list)
    for _, row in intervals_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        key = (str(row["chr"]), str(row["gene"]))
        index[key].append((start, end, float(row[score_col])))

    for intervals in index.values():
        intervals.sort(key=lambda x: (x[0], x[1]))
    return dict(index)


def iter_overlapping_intervals(
    intervals: Iterable[Interval],
    query_start: int,
    query_end: int,
) -> Iterable[Interval]:
    for start, end, score in intervals:
        if end <= query_start:
            continue
        if start >= query_end:
            break
        yield start, end, score


def assign_max_overlap_scores_by_gene(
    peaks_df: pd.DataFrame,
    source_df: pd.DataFrame,
    score_col: str = "score",
) -> pd.Series:
    interval_index = build_interval_index_by_gene(source_df, score_col=score_col)
    scores = []
    for _, row in peaks_df.iterrows():
        key = (str(row["chr"]), str(row["gene"]))
        overlaps = iter_overlapping_intervals(
            interval_index.get(key, []),
            int(row["start"]),
            int(row["end"]),
        )
        scores.append(max((score for _, _, score in overlaps), default=0.0))
    return pd.Series(scores, index=peaks_df.index, dtype=float)


def merge_intervals_by_gene(peaks_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["chr", "start", "end", "gene", "sources", "max_score"]
    if len(peaks_df) == 0:
        return pd.DataFrame(columns=columns)

    peaks_df = peaks_df.copy()
    peaks_df["chr"] = peaks_df["chr"].astype(str)
    peaks_df["gene"] = peaks_df["gene"].astype(str)
    peaks_df["start"] = pd.to_numeric(peaks_df["start"], errors="coerce")
    peaks_df["end"] = pd.to_numeric(peaks_df["end"], errors="coerce")
    peaks_df["score"] = pd.to_numeric(peaks_df["score"], errors="coerce").fillna(0.0)
    peaks_df = peaks_df.dropna(subset=["start", "end"])
    peaks_df["start"] = peaks_df["start"].astype(int)
    peaks_df["end"] = peaks_df["end"].astype(int)
    peaks_df = peaks_df[peaks_df["end"] > peaks_df["start"]]

    merged_rows = []
    for (chrom, gene), group in peaks_df.groupby(["chr", "gene"], sort=True):
        group = group.sort_values(["start", "end"])
        current = None
        for _, row in group.iterrows():
            if current is None or int(row["start"]) > current["end"]:
                if current is not None:
                    merged_rows.append(current)
                current = {
                    "chr": chrom,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "gene": gene,
                    "sources": {str(row["source"])},
                    "max_score": float(row["score"]),
                }
                continue

            current["end"] = max(current["end"], int(row["end"]))
            current["sources"].add(str(row["source"]))
            current["max_score"] = max(current["max_score"], float(row["score"]))

        if current is not None:
            merged_rows.append(current)

    for row in merged_rows:
        row["sources"] = ",".join(sorted(row["sources"]))
    return pd.DataFrame(merged_rows, columns=columns)
