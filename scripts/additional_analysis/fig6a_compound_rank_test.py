#!/usr/bin/env python
"""A7 / R1 M6,m7–8: tie-aware test of frozen compound groups (no API calls)."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from _common import read_tsv, unique, save, output_dir, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Frozen TSV: compound,group,rank,included,exclusion_reason,annotation_evidence")
    p.add_argument("--group-history", required=True, help="Author record of selection timing or 'unknown'")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = read_tsv(a.input, ["compound", "group", "rank", "included", "exclusion_reason", "annotation_evidence"])
    unique(df, ["compound"])
    if not df.included.isin([0, 1]).all():
        raise ValueError("included must be 0 or 1")
    keep = df[df.included == 1]
    if not keep.group.isin(["corticosteroid", "other"]).all():
        raise ValueError("Included groups must be corticosteroid or other")
    if not np.isfinite(keep['rank']).all() or not keep['rank'].between(0, 1).all():
        raise ValueError("Included rank must be finite in [0,1]; missing input is not non-detection")
    if df.loc[df.included == 0, "exclusion_reason"].isna().any():
        raise ValueError("Every excluded compound needs a reason")
    x, y = [keep.loc[keep.group == g, "rank"].to_numpy() for g in ["corticosteroid", "other"]]
    if not len(x) or not len(y):
        raise ValueError("Both fixed groups must contain compounds")
    rows = []
    for alternative in ["greater", "two-sided"]:
        result = mannwhitneyu(x, y, alternative=alternative, method="asymptotic", use_continuity=True)
        # The all-tied case has no rank evidence for either direction.
        pvalue = 1. if np.ptp(np.concatenate([x, y])) == 0 else result.pvalue
        rows.append(dict(alternative=alternative, n_corticosteroid=len(x), n_other=len(y),
                         n_zero_corticosteroid=int((x == 0).sum()), n_zero_other=int((y == 0).sum()),
                         u=result.statistic, pvalue=pvalue,
                         probability_superiority=result.statistic/(len(x)*len(y)),
                         interpretation="biologically motivated exploratory test"))
    out = output_dir(a.out)
    save(df, out, "compound_group_audit.tsv")
    save(pd.DataFrame(rows), out, "nr3c1_tie_aware_tests.tsv")
    provenance(out, a, [a.input])


if __name__ == "__main__":
    main()
