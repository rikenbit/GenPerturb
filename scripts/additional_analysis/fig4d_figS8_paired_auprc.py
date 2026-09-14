#!/usr/bin/env python
"""A3 / R1 M4; R2 M5: joint-method paired perturbation bootstrap."""
import argparse
import numpy as np
import pandas as pd
from _common import read_tsv, unique, save, output_dir, provenance


def summarize(df, reference, repeats, seed):
    unique(df, ["universe", "perturbation", "stratum", "method"])
    rng = np.random.default_rng(seed)
    points, summaries = [], []
    for (universe, stratum), group in df.groupby(["universe", "stratum"], sort=True):
        wide = group.pivot(index="perturbation", columns="method", values="auprc")
        if reference not in wide:
            raise ValueError(f"Missing reference {reference}")
        # Use the common complete perturbation set for all methods in this stratum.
        wide = wide.replace([np.inf, -np.inf], np.nan).dropna()
        draw = rng.integers(len(wide), size=(repeats, len(wide))) if len(wide) >= 2 else None
        for method in wide.columns.drop(reference):
            delta = (wide[reference] - wide[method]).to_numpy()
            ci = np.quantile(delta[draw].mean(axis=1), [.025, .975]) if draw is not None else [np.nan]*2
            summaries.append(dict(universe=universe, stratum=stratum, comparator=method, n=len(delta),
                                  mean_difference=delta.mean() if len(delta) else np.nan,
                                  ci_low=ci[0], ci_high=ci[1], status="ok" if len(delta)>=2 else "insufficient_perturbations"))
            for pert, d in zip(wide.index, delta):
                points.append(dict(universe=universe, stratum=stratum, comparator=method,
                                   perturbation=pert, difference=d))
    return pd.DataFrame(points), pd.DataFrame(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default="GenPerturb")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.bootstrap < 1:
        p.error("--bootstrap must be positive")
    df = read_tsv(a.input, ["universe", "perturbation", "stratum", "method", "auprc"])
    points, summary = summarize(df, a.reference, a.bootstrap, a.seed)
    out = output_dir(a.out)
    save(points, out, "paired_points.tsv")
    save(summary, out, "paired_auprc_summary.tsv")
    provenance(out, a, [a.input])


if __name__ == "__main__":
    main()
