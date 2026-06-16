#!/bin/bash
# ==============================================================================
# Fig 2 / S1-S7: UMAP / clustering 
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig2_24_clustering_signature
pipeline_activate_conda singlecell

python scripts/expression_model_performance/21_embedding_signature.py
python scripts/expression_model_performance/22_compare_model_clustering.py
