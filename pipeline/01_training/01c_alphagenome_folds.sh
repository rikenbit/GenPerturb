#!/bin/bash
# ==============================================================================
# 01c: AlphaGenome fold_0..3 transfer (heavy GPU)
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

DISPATCHER="${PROJECT_ROOT}/pipeline/01_slurm.sh"

FOLD_STUDIES=(
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "JialongJiang2024_CD8T_train"
)
FOLD_SHORT=("norman" "jj_cd8t")

echo "=== AlphaGenome fold_0..3 transfer (${#FOLD_STUDIES[@]} studies x 4 folds) ==="
for i in "${!FOLD_STUDIES[@]}"; do
    for FOLD in 0 1 2 3; do
        sbatch -J "ag_f${FOLD}_${FOLD_SHORT[$i]}" \
            -o "log/01c_ag_fold${FOLD}_${FOLD_SHORT[$i]}.out" \
            -e "log/01c_ag_fold${FOLD}_${FOLD_SHORT[$i]}.err" \
            --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
            "${DISPATCHER}" -p model "${FOLD_STUDIES[$i]}" "transfer" "alphagenome_fold_${FOLD}"
    done
done
