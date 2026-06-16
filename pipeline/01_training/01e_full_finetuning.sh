#!/bin/bash
# ==============================================================================
# 01e: AlphaGenome full finetuning (heavy GPU, long-running)
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

DISPATCHER="${PROJECT_ROOT}/pipeline/01_slurm.sh"
NODELIST="${NODELIST:-gpu03}"

TRAINING_STUDIES=(
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "JialongJiang2024_CD8T_train"
)
TRAINING_SHORT=("norman" "jj_cd8t")

echo "=== AlphaGenome full finetuning (${#TRAINING_STUDIES[@]} studies) ==="
for i in "${!TRAINING_STUDIES[@]}"; do
    sbatch -J "ag_ft_${TRAINING_SHORT[$i]}" \
        -o "log/01e_ag_finetuning_${TRAINING_SHORT[$i]}.out" \
        -e "log/01e_ag_finetuning_${TRAINING_SHORT[$i]}.err" \
        --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
        --time=20-00:00:00 --nodelist="${NODELIST}" \
        "${DISPATCHER}" -p model "${TRAINING_STUDIES[$i]}" "finetuning" "alphagenome"
done
