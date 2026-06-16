#!/usr/bin/env bash
# ==============================================================================
# Fig S8 / 50: all-gene attribution-axis controls (Martin)
# ==============================================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig3_50_attribution_axis_allgenes
pipeline_activate_conda singlecell

python scripts/attribution_evaluation/61_compute_attribution_axis_allgenes.py --study Martin
python scripts/attribution_evaluation/62_make_selected_martin_attribution_figures.py
