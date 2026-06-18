#!/bin/bash
# ==============================================================================
# 01d: LoRA training sweep (heavy GPU, long-running)
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

DISPATCHER="${PROJECT_ROOT}/pipeline/01_slurm.sh"
LORA_ALPHA="${LORA_ALPHA:-2}"
LORA_RANKS="${LORA_RANKS:-64 256 512}"
NODELIST="${NODELIST:-}"

TRAINING_STUDIES=(
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "JialongJiang2024_CD8T_train"
)
TRAINING_SHORT=("norman" "jj_cd8t")

echo "=== AlphaGenome LoRA sweep (ranks=${LORA_RANKS}, alpha=${LORA_ALPHA}) ==="
for i in "${!TRAINING_STUDIES[@]}"; do
    for LORA_R in ${LORA_RANKS}; do
        SBATCH_ARGS=(
            -J "ag_lora_r${LORA_R}_${TRAINING_SHORT[$i]}"
            -o "log/01d_ag_lora_r${LORA_R}_${TRAINING_SHORT[$i]}.out" \
            -e "log/01d_ag_lora_r${LORA_R}_${TRAINING_SHORT[$i]}.err" \
            --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
            --time=20-00:00:00
        )
        [ -n "$NODELIST" ] && SBATCH_ARGS+=(--nodelist="${NODELIST}")
        sbatch "${SBATCH_ARGS[@]}" \
            "${DISPATCHER}" -p model "${TRAINING_STUDIES[$i]}" "lora" "alphagenome" "${LORA_R}" "${LORA_ALPHA}"
    done
done
