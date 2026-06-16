#!/bin/bash
# ==============================================================================
# Fig 3 / 33: Paper motif summaries for Fig 3e / Fig S10a
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"

pipeline_start_log "fig3_33_motif_summary_${STUDY%%_*}"

pipeline_activate_conda gimme
python scripts/attribution_evaluation/32_summary_gimmemotifs.py "$STUDY" "$STUDY_SUFFIX"

pipeline_activate_conda modisco
python scripts/attribution_evaluation/34_summary_tfmodisco.py "$STUDY" "$STUDY_SUFFIX"

pipeline_activate_conda gimme
python scripts/attribution_evaluation/36_summary_combined_motif.py "$STUDY" "$STUDY_SUFFIX"
