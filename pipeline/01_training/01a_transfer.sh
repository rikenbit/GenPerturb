#!/bin/bash
# ==============================================================================
# 01a: Transfer learning (heavy GPU, one job per study x model)
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${DEFAULT_MODELS[@]}")

if [ -n "${STUDIES_OVERRIDE:-}" ]; then
    read -r -a STUDIES <<< "${STUDIES_OVERRIDE}"
    SHORT=()
    for s in "${STUDIES[@]}"; do SHORT+=("${s%%_*}"); done
fi

DISPATCHER="${PROJECT_ROOT}/pipeline/01_slurm.sh"

for model in "${MODELS[@]}"; do
    case "$model" in
        alphagenome) prefix="ag" ;;
        borzoi)      prefix="bz" ;;
        enformer)    prefix="en" ;;
        *)           prefix="${model:0:2}" ;;
    esac

    echo "=== Submitting ${model} transfer (${#STUDIES[@]} jobs) ==="
    for i in "${!STUDIES[@]}"; do
        sbatch -J "${prefix}_tr_${SHORT[$i]}" \
            -o "log/01a_${model}_transfer_${SHORT[$i]}.out" \
            -e "log/01a_${model}_transfer_${SHORT[$i]}.err" \
            --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
            "${DISPATCHER}" -p model "${STUDIES[$i]}" "transfer" "${model}"
    done
done
