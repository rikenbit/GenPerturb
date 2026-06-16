#!/bin/bash
# ==============================================================================
# 00b: Assemble training tables from embeddings
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log 00b_build_training
pipeline_activate_conda singlecell

mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

for VERSION in all_folds fold_0 fold_1 fold_2 fold_3; do
    echo "=== 30_process_training_dataset.py (${VERSION}) ==="
    python genperturb/preprocess/30_process_training_dataset.py --model_version ${VERSION}
done

echo "=== 32_match_training_dataset.py ==="
python genperturb/preprocess/32_match_training_dataset.py
