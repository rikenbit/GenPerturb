#!/bin/bash
# ==============================================================================
# 00d: Baseline data tables
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log 00d_baseline_data
pipeline_activate_conda singlecell

mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

python genperturb/preprocess/15_create_baselinedata.py
