#!/bin/bash
# ==============================================================================
# Fig. 4e / S9a: root-output TF-MoDISco for the motif-discovery route
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"

pipeline_start_log "fig3_32_run_tfmodisco_${STUDY%%_*}"
pipeline_activate_conda modisco

python scripts/attribution_evaluation/33_run_tfmodisco.py \
    "${STUDY}" "${STUDY_SUFFIX}"
