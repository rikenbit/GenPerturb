#!/bin/bash
# ==============================================================================
# Fig 4 / 21: Aggregate GimmeMotifs (light CPU)
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

pipeline_start_log "fig4_21_gimme_agg_${STUDY%%_*}"

pipeline_activate_conda gimme
python scripts/attribution_analysis/08-aggregate_gimmemotifs.py \
    --study "${STUDY}" \
    --study-suffix "${STUDY_SUFFIX}" \
    --input-base "${INPUT_BASE}" \
    --output-base "${INPUT_BASE}"

