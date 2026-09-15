#!/bin/bash
# ==============================================================================
# 71: Method-independent candidate AUPRC and paired bootstrap
#   Fig. 4d, Fig. S8a/b
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

OUTDIR="${OUTDIR:-results/enhancer_benchmark}"
MANIFEST="${MANIFEST:-scripts/additional_analysis/auprc_manifest.tsv}"
MISSING_ATTRIBUTION_POLICY="${MISSING_ATTRIBUTION_POLICY:-zero}"

pipeline_start_log step71_enhancer_benchmark
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/additional_analysis/fig4d_figS8_independent_auprc.py \
    --manifest "${MANIFEST}" \
    --genes data/science.ads7951_tables_s1_to_s6/science.ads7951_table_s3.xlsx \
    --tss-bed fasta/MartinRufino2025_mixscape_exnp_train.bed \
    --chrom-sizes fasta/GRCh38.p14.genome.fa.sizes \
    --context-length 1048576 \
    --missing-attribution-policy "${MISSING_ATTRIBUTION_POLICY}" \
    --out "${OUTDIR}/output"

python scripts/additional_analysis/fig4d_figS8_paired_auprc.py \
    --input "${OUTDIR}/output/auprc_per_perturbation.tsv" \
    --out "${OUTDIR}/paired_comparison"
