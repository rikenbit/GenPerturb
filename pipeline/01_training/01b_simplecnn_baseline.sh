#!/bin/bash
# ==============================================================================
# 01b: SimpleCNN baseline (heavy GPU, batched dependency)
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

BATCH_SIZE="${BATCH_SIZE:-4}"
DISPATCHER="${PROJECT_ROOT}/pipeline/01_slurm.sh"

if [ -n "${STUDIES_OVERRIDE:-}" ]; then
    read -r -a STUDIES <<< "${STUDIES_OVERRIDE}"
    SHORT=()
    for s in "${STUDIES[@]}"; do SHORT+=("${s%%_*}"); done
fi

echo "=== Submitting SimpleCNN baseline (${#STUDIES[@]} jobs, batch=${BATCH_SIZE}) ==="

CNN_JIDS=()
BATCH_DEP=""
for i in "${!STUDIES[@]}"; do
    if (( i > 0 && i % BATCH_SIZE == 0 )); then
        BATCH_DEP="--dependency=afterany:$(IFS=:; echo "${CNN_JIDS[*]}")"
        CNN_JIDS=()
    fi
    JID=$(sbatch --parsable $BATCH_DEP \
        -J "cnn_${SHORT[$i]}" \
        -o "log/01b_cnn_${SHORT[$i]}.out" \
        -e "log/01b_cnn_${SHORT[$i]}.err" \
        --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" --time=7-00:00:00 \
        "${DISPATCHER}" -p model "${STUDIES[$i]}" "baseline" "simplecnn")
    CNN_JIDS+=("$JID")
    echo "  cnn_${SHORT[$i]}: ${JID}${BATCH_DEP:+ dep: ${BATCH_DEP}}"
done
