#!/bin/bash
# ==============================================================================
# Fig 3: root-output peak call after paper Captum raw H5 generation
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:?Usage: $0 STUDY MODEL [SUFFIX] [H5_SUFFIX]}"
MODEL="${2:?Usage: $0 STUDY MODEL [SUFFIX] [H5_SUFFIX]}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"
H5_SUFFIX="${4:-}"

pipeline_start_log "fig3_12_peak_call_${STUDY}"
pipeline_activate_conda "$MODEL"

ARGS=("$STUDY" "$STUDY_SUFFIX" variable_genes)
if [ -n "$H5_SUFFIX" ]; then
    ARGS+=(--h5-suffix "$H5_SUFFIX")
fi

python scripts/attribution_evaluation/10a_peak_call.py "${ARGS[@]}"
