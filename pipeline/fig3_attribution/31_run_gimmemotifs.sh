#!/bin/bash
# ==============================================================================
# Fig 3e / S10a: root-output GimmeMotifs for the paper motif-discovery route
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"

pipeline_start_log "fig3_31_run_gimmemotifs_${STUDY%%_*}"
pipeline_activate_conda gimme

python scripts/attribution_evaluation/31_run_gimmemotifs.py \
    "${STUDY}" "${STUDY_SUFFIX}"
