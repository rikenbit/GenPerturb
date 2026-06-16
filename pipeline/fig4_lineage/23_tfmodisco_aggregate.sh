#!/bin/bash
# ==============================================================================
# Fig 4 / 23: Aggregate TF-MoDISco (light CPU)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-NormanWeissman2019_filtered_mixscape_exnp_train}"
STUDY_SUFFIX="${2:-$(default_study_suffix alphagenome)}"
WITH_PLOTS="${3:-}"
INPUT_BASE="attribution_analysis"
ADATA_DIR="adata"
TOP_N_PER_CLUSTER=10
QVAL_THRESHOLD=1.0
MATRIX_TYPES="pos neg signed"

pipeline_start_log "fig4_23_modisco_agg_${STUDY%%_*}"
pipeline_activate_conda singlecell

python scripts/attribution_analysis/08b-aggregate_tfmodisco.py \
    --study "${STUDY}" \
    --study-suffix "${STUDY_SUFFIX}" \
    --input-base "${INPUT_BASE}" \
    --output-base "${INPUT_BASE}" \
    --qval-threshold "${QVAL_THRESHOLD}"


