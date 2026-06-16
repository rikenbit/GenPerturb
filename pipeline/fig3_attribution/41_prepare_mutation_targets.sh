#!/bin/bash
# ==============================================================================
# Fig 3f / 41: Prepare mutation targets before 42_run_mutation_array.sh
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY_KEY="${1:-Martin_matched}"
shift $(( $# >= 1 ? 1 : 0 ))

pipeline_start_log "fig3_41_prepare_mutation_targets_${STUDY_KEY}"
pipeline_activate_conda singlecell

if [[ "$STUDY_KEY" == *_matched ]]; then
    python scripts/attribution_evaluation/52a_matched_prepare_mutation_targets.py --study "$STUDY_KEY"
else
    python scripts/attribution_evaluation/52a_prepare_mutation_targets.py --study "$STUDY_KEY" "$@"
fi
