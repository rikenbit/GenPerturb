#!/bin/bash
# ==============================================================================
# Fig 2 / S1-S7: Compare training methods
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig2_23_compare_training
pipeline_activate_conda singlecell

python scripts/expression_model_performance/02_compare_training.py
python scripts/expression_model_performance/05_compare_ptft_realobs.py
