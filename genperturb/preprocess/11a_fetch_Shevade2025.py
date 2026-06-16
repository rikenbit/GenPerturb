#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("_common_11a", Path(__file__).with_name("11a_common.py"))
_common = module_from_spec(_spec)
_spec.loader.exec_module(_common)  # type: ignore[union-attr]
DATA_DIR = _common.DATA_DIR
download = _common.download

GSE = "GSE288996"
SAMPLE_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples"
DEST = DATA_DIR / GSE


def per_gsm_url(fn: str) -> str:
    gsm = fn.split("_", 1)[0]
    stem = gsm[:-3] + "nnn"
    return f"{SAMPLE_BASE}/{stem}/{gsm}/suppl/{fn}"


FILES = [
    ("GSM8780553_iPSC_ATAC_atac_fragments.tsv.gz", 1_881_467_752),
    ("GSM8780553_iPSC_ATAC_barcodes.tsv.gz", 40_744),
    ("GSM8780553_iPSC_ATAC_matrix.mtx.gz", 308_307_842),
    ("GSM8780554_iPSC_GEX_barcodes.tsv.gz", 40_744),
    ("GSM8780554_iPSC_GEX_features.tsv.gz", 6_022_892),
    ("GSM8780554_iPSC_GEX_matrix.mtx.gz", 308_307_842),
    ("GSM8780555_iPSC_guideRNA.txt.gz", 144),
    ("GSM8780556_K562_Dasatinib_ATAC_1_atac_fragments.tsv.gz", 2_898_121_199),
    ("GSM8780556_K562_Dasatinib_ATAC_1_barcodes.tsv.gz", 58_817),
    ("GSM8780556_K562_Dasatinib_ATAC_1_matrix.mtx.gz", 412_119_086),
    ("GSM8780557_K562_Dasatinib_ATAC_2_atac_fragments.tsv.gz", 2_576_418_529),
    ("GSM8780557_K562_Dasatinib_ATAC_2_barcodes.tsv.gz", 76_309),
    ("GSM8780557_K562_Dasatinib_ATAC_2_matrix.mtx.gz", 422_349_197),
    ("GSM8780558_K562_DMSO_ATAC_1_atac_fragments.tsv.gz", 2_654_349_395),
    ("GSM8780558_K562_DMSO_ATAC_1_barcodes.tsv.gz", 62_115),
    ("GSM8780558_K562_DMSO_ATAC_1_matrix.mtx.gz", 469_874_842),
    ("GSM8780559_K562_DMSO_ATAC_2_atac_fragments.tsv.gz", 2_510_891_274),
    ("GSM8780559_K562_DMSO_ATAC_2_barcodes.tsv.gz", 66_231),
    ("GSM8780559_K562_DMSO_ATAC_2_matrix.mtx.gz", 499_174_805),
    ("GSM8780560_K562_Dasatinib_guideRNA_1.txt.gz", 455),
    ("GSM8780561_K562_Dasatinib_guideRNA_2.txt.gz", 464),
    ("GSM8780562_K562_DMSO_guideRNA_1.txt.gz", 449),
    ("GSM8780563_K562_DMSO_guideRNA_2.txt.gz", 448),
    ("GSM8780564_K562_Dasatinib_RNA_1_barcodes.tsv.gz", 58_817),
    ("GSM8780564_K562_Dasatinib_RNA_1_features.tsv.gz", 4_219_737),
    ("GSM8780564_K562_Dasatinib_RNA_1_matrix.mtx.gz", 412_119_086),
    ("GSM8780565_K562_Dasatinib_RNA_2_barcodes.tsv.gz", 76_309),
    ("GSM8780565_K562_Dasatinib_RNA_2_features.tsv.gz", 4_244_060),
    ("GSM8780565_K562_Dasatinib_RNA_2_matrix.mtx.gz", 422_349_197),
    ("GSM8780566_K562_DMSO_RNA_1_barcodes.tsv.gz", 62_115),
    ("GSM8780566_K562_DMSO_RNA_1_features.tsv.gz", 4_931_033),
    ("GSM8780566_K562_DMSO_RNA_1_matrix.mtx.gz", 469_874_842),
    ("GSM8780567_K562_DMSO_RNA_2_barcodes.tsv.gz", 66_231),
    ("GSM8780567_K562_DMSO_RNA_2_features.tsv.gz", 4_748_584),
    ("GSM8780567_K562_DMSO_RNA_2_matrix.mtx.gz", 499_174_805),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="skip ATAC fragments (~14 GB)")
    ap.add_argument("--k562-dmso", action="store_true",
                    help="restrict to K562 DMSO RNA + guideRNA trios (default for GenPerturb)")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="substring match on filenames (e.g. iPSC, K562_DMSO)")
    ap.add_argument("--list", action="store_true", help="list and exit")
    args = ap.parse_args(argv)

    DEST.mkdir(parents=True, exist_ok=True)

    picked = []
    for fn, sz in FILES:
        if args.light and "atac_fragments" in fn:
            continue
        if args.k562_dmso:
            if "K562_DMSO" not in fn:
                continue
            if "_ATAC" in fn:  # skip ATAC trio; keep guideRNA and RNA only
                continue
        if args.samples and not any(s in fn for s in args.samples):
            continue
        picked.append((fn, sz))

    total = sum(s for _, s in picked) / 1e9
    print(f"[plan] {len(picked)} files, ~{total:.1f} GB")
    if args.list:
        for fn, sz in picked:
            print(f"  {sz/1e6:>9.1f} MB  {fn}")
        return 0

    for fn, sz in picked:
        download(per_gsm_url(fn), DEST / fn, sz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
