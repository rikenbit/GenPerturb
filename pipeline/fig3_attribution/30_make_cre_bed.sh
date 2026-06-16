#!/bin/bash
# ==============================================================================
# Fig 3e / S10a / 30: CRE BED creation (light CPU)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"

pipeline_start_log "fig3_30_cre_${STUDY%%_*}"
pipeline_activate_conda alphagenome

python scripts/attribution_evaluation/21_make_cre_bed.py \
    --study_name "${STUDY}" \
    --pretrained_model "${MODEL}" \
    --study_suffix "${STUDY_SUFFIX}"
