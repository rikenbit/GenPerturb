#!/bin/bash
# ==============================================================================
# Fig 2 / S1-S7: Compare models across studies 
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig2_21_compare_models
pipeline_activate_conda singlecell

python scripts/expression_model_performance/01_compare_models.py
python scripts/expression_model_performance/01a_compare_models_across_genes.py
python scripts/expression_model_performance/01b_compare_models_across_perts.py