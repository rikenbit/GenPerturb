#!/bin/bash
# ==============================================================================
# 75: High-attribution tertile panels for Fig. 4f and Fig. S9b
#   Consumes mutation_pairs.tsv from 72; writes the tertile_panels/ inputs of 78
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

RESULTS="${RESULTS:-results/mutation_effects}"
FIGURES="${FIGURES:-figures/mutation_effects}"
TERTILE="${TERTILE:-${RESULTS}/tertile_panels}"

pipeline_start_log step75_tertile_panels
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for STUDY in Martin Norman; do
    python scripts/additional_analysis/fig4f_figS9b_mutation_tertile_panel.py \
        --pairs "${RESULTS}/${STUDY}_matched/mutation_pairs.tsv" \
        --label "${STUDY}" \
        --out "${TERTILE}/${STUDY}_tertile" \
        --figure-out "${FIGURES}/tertile/${STUDY}_tertile"
done
