#!/bin/bash
# ==============================================================================
# Fig 3f / 40: Prepare seqlet metadata for motif mutation analyses
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

QVAL="${1:-0.05}"

pipeline_start_log fig3_40_prepare_seqlet_metadata
pipeline_activate_conda singlecell

python scripts/attribution_evaluation/51a_create_union_gene_list.py
python scripts/attribution_evaluation/51b_extract_seqlet_metadata.py --qval "$QVAL"
python scripts/attribution_evaluation/51c_add_motif_core.py
