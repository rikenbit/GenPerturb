#!/bin/bash
# ==============================================================================
# Fig 2 / S1-S7: Multifold + baselines
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig2_22_multifold_baseline
pipeline_activate_conda singlecell

python scripts/expression_model_performance/03_compare_multifold.py
python scripts/expression_model_performance/04_compare_baselines.py
