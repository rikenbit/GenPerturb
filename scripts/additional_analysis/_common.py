"""Small, portable utilities for revision analyses (no project-root assumptions)."""
import hashlib
import json
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd


def read_tsv(path, required=()):
    df = pd.read_csv(path, sep="\t")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def unique(df, keys):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate identifiers: {keys}")


def output_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def save(df, out, name):
    df.to_csv(out / name, sep="\t", index=False, na_rep="NA")


def provenance(out, args, inputs):
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "h5py", "anndata"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    files = []
    for path in inputs:
        p = Path(path).resolve()
        entry = {"path": str(p), "exists": p.is_file()}
        if p.is_file():
            entry.update(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            # Hash small metadata; large arrays are identified by path/size/mtime.
            if p.stat().st_size <= 10_000_000:
                entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append(entry)
    (out / "run.json").write_text(json.dumps({
        "arguments": vars(args), "command": sys.argv, "packages": packages,
        "inputs": files,
    }, indent=2, default=str) + "\n")


def bootstrap_mean(values, seed=42, repeats=10000):
    x = np.asarray(values, dtype=float)
    if repeats < 1 or not np.isfinite(x).all():
        raise ValueError("Bootstrap needs finite values and positive repeats")
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(repeats)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)
