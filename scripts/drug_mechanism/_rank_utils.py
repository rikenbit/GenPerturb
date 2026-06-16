from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_percentile_matrix(
    score_df: pd.DataFrame,
    exclude_zero: bool = False,
    method: str = "average",
) -> pd.DataFrame:
    out = np.full(score_df.shape, np.nan if exclude_zero else 0.0, dtype=float)
    arr = score_df.values
    for j in range(score_df.shape[1]):
        col = arr[:, j].astype(float)
        mask = np.isfinite(col)
        if exclude_zero:
            mask &= col > 0
        n = int(mask.sum())
        if n == 0:
            continue
        if n == 1:
            out[mask, j] = 1.0
            continue
        ranks_asc = rankdata(col[mask], method=method)
        out[mask, j] = (ranks_asc - 1.0) / (n - 1.0)
    return pd.DataFrame(out, index=score_df.index, columns=score_df.columns)
