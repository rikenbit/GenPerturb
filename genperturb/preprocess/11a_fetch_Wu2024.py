#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("_common_11a", Path(__file__).with_name("11a_common.py"))
_common = module_from_spec(_spec)
_spec.loader.exec_module(_common)  # type: ignore[union-attr]
DATA_DIR = _common.DATA_DIR
download = _common.download
untar = _common.untar

GSE = "GSE277747"
DEST = DATA_DIR / GSE
TAR_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE277nnn/{GSE}/suppl/{GSE}_RAW.tar"
TAR_BYTES = 70_717_440


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-convert", action="store_true",
                    help="skip the Rscript SCE -> mtx conversion step")
    ap.add_argument("--r-env", default="rconv277",
                    help="conda env name containing Rscript + bioconductor-* (default: rconv277)")
    args = ap.parse_args(argv)

    DEST.mkdir(parents=True, exist_ok=True)
    tar_path = DEST / f"{GSE}_RAW.tar"
    download(TAR_URL, tar_path, TAR_BYTES)

    raw_dir = DEST / "raw"
    untar(tar_path, raw_dir)

    for gz in raw_dir.glob("*.RDS.gz"):
        out = gz.with_suffix("")
        if not out.exists():
            print(f"[gzip] {gz.name} -> {out.name}")
            with gzip.open(gz, "rb") as g, open(out, "wb") as f:
                shutil.copyfileobj(g, f, length=1 << 20)

    if args.no_convert:
        return 0

    rds = next(raw_dir.glob("*.RDS"), None)
    if rds is None:
        print("[err ] no RDS found in raw/")
        return 1

    out_dir = DEST / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = Path(__file__).with_name("11a_convert_Wu2024_sce.R")

    r_bin = shutil.which("Rscript")
    if r_bin is not None:
        subprocess.run([r_bin, str(conv), str(rds), str(out_dir)], check=True)
        return 0

    # Fallback: run via conda env
    conda = shutil.which("conda")
    if conda is None:
        print("[err ] neither Rscript nor conda on PATH. "
              f"Create the env first (e.g. 'conda create -n {args.r_env} -c conda-forge -c bioconda "
              "r-base bioconductor-singlecellexperiment bioconductor-summarizedexperiment "
              "bioconductor-s4vectors bioconductor-genomicranges r-matrix').")
        return 2
    subprocess.run([conda, "run", "-n", args.r_env, "Rscript",
                    str(conv), str(rds), str(out_dir)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
