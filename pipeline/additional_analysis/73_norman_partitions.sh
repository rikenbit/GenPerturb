#!/bin/bash
# ==============================================================================
# 73: Norman observed and fitted partitions
#   Fig. 5a source panels
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

OUTDIR="${OUTDIR:-revision_analysis/results/A6_corrected_v3}"
FIGDIR="${FIGDIR:-revision_analysis/figures/norman_partitions}"

pipeline_start_log revision_73_norman_partitions
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/additional_analysis/fig5a_norman_partitions.py \
    --root "${PROJECT_ROOT}" --out "${OUTDIR}" --figure-out "${FIGDIR}"
