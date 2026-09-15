#!/bin/bash
# ==============================================================================
# 78: Draw publication-sized panels
#   Fig. 4d, Fig. 4f, Fig. S8a/b, Fig. S9b, placed at 1:1 by the figure scripts
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

RESULTS="${RESULTS:-results}"
FIGURES="${FIGURES:-figures}"
OUTDIR="${1:-${RESULTS}/print_panels}"

pipeline_start_log step78_print_size_panels
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/additional_analysis/fig4_figS8_figS9_print_panels.py \
    --auprc-root "${RESULTS}/enhancer_benchmark" \
    --martin-tertile "${RESULTS}/mutation_effects/tertile_panels/Martin_tertile" \
    --norman-tertile "${RESULTS}/mutation_effects/tertile_panels/Norman_tertile" \
    --out "${OUTDIR}" \
    --figure-out "${FIGURES}/print_panels"
