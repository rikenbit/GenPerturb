#!/bin/bash
# ==============================================================================
# Fig 4 / 32: Seqlet long table + cancellation plots (S11)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig4_32_seqlet_cancellation
pipeline_activate_conda singlecell

python scripts/immune_differentiation/22_build_seqlet_long.py

python scripts/immune_differentiation/23_signature_cancellation_plot.py
