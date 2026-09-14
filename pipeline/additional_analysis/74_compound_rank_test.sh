#!/bin/bash
# ==============================================================================
# 74: Tie-aware test of the frozen compound groups
#   Two-sided p value annotated onto Fig. 6a; no PubChem re-query
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

INPUT="${INPUT:-revision_analysis/results/A7_compound_v2/compound_groups.tsv}"
OUTDIR="${OUTDIR:-revision_analysis/results/A7_compound_v2/tie_aware}"

pipeline_start_log revision_74_compound_rank_test
pipeline_activate_conda singlecell

python scripts/additional_analysis/fig6a_compound_rank_test.py \
    --input "${INPUT}" --group-history unknown --out "${OUTDIR}"
