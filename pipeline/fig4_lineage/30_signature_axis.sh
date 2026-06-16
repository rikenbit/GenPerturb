#!/bin/bash
# ==============================================================================
# Fig 4 / 30: Signature axis plots (S11)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig4_30_signature_axis
pipeline_activate_conda singlecell

python scripts/immune_differentiation/01_gene_signature.py
python scripts/immune_differentiation/11_signature_axis_plots.py
