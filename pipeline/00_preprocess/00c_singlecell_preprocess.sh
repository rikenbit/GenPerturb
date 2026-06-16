#!/bin/bash
# ==============================================================================
# 00c: Single-cell preprocessing 
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log 00c_singlecell_preprocess
pipeline_activate_conda singlecell

mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

echo "=== 11_preprocess_adata.py ==="
python genperturb/preprocess/11_preprocess_adata.py

echo "=== 12_pertpy.py ==="
python genperturb/preprocess/12_pertpy.py

echo "=== 13_prepare_pseudobulk.py ==="
python genperturb/preprocess/13_prepare_pseudobulk.py

python genperturb/preprocess/16_prepare_atac_pseudobulk.py

