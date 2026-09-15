#!/bin/bash
# ==============================================================================
# 74: Tie-aware test of the frozen compound groups
#   Two-sided p value annotated onto Fig. 6a; no PubChem re-query
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

CD8T_FULL="JialongJiang2024_CD8T_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
INPUT="${INPUT:-figures/${CD8T_FULL}/drug_mechanism/compound_groups.tsv}"
OUTDIR="${OUTDIR:-results/nr3c1_condition_test}"

pipeline_start_log step74_compound_rank_test
pipeline_activate_conda singlecell

python scripts/additional_analysis/fig6a_compound_rank_test.py \
    --input "${INPUT}" --out "${OUTDIR}"
