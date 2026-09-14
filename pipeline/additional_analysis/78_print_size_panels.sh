#!/bin/bash
# ==============================================================================
# 78: Redraw the revised panels at their final printed size
#   Fig. 4d, Fig. 4f, Fig. S8a/b, Fig. S9b, placed at 1:1 by the figure scripts
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

RESULTS="${RESULTS:-revision_analysis/results}"
FIGURES="${FIGURES:-revision_analysis/figures}"
OUTDIR="${1:-${RESULTS}/print_panels_right_only}"

pipeline_start_log revision_78_print_size_panels
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/additional_analysis/fig4_figS8_figS9_print_panels.py \
    --auprc-root "${RESULTS}/P2_A2_complete_assumption" \
    --martin-tertile "${RESULTS}/tertile_panels/Martin_tertile" \
    --norman-tertile "${RESULTS}/tertile_panels/Norman_tertile" \
    --out "${OUTDIR}" \
    --figure-out "${FIGURES}/print_panels"
