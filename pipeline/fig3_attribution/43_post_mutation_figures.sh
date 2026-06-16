#!/bin/bash
# ==============================================================================
# Fig 3f / 43: Local post-processing after 42_run_mutation_array.sh
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY_KEY="${1:-Martin_matched}"

pipeline_start_log "fig3_43_post_mutation_${STUDY_KEY}"

if [ "$STUDY_KEY" = "Martin_full" ]; then
    pipeline_activate_conda singlecell
    python scripts/attribution_evaluation/52c_build_tables.py --study "$STUDY_KEY"
    python scripts/attribution_evaluation/53a_select_candidates.py

    pipeline_activate_conda modisco
    python scripts/attribution_evaluation/53b_plot_paper_figure.py
elif [[ "$STUDY_KEY" == *_matched ]]; then
    pipeline_activate_conda singlecell
    python scripts/attribution_evaluation/52d_matched_plot_attr_magnitude_boxplot.py --study "$STUDY_KEY"
else
    pipeline_activate_conda singlecell
    python scripts/attribution_evaluation/52c_build_tables.py --study "$STUDY_KEY"
fi
