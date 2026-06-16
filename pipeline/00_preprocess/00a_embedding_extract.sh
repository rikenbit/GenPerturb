#!/bin/bash
# ==============================================================================
# 00a: Pretrained-model embedding extraction 
# ==============================================================================


set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("alphagenome")

submit_embedding() {
    local model="$1"
    local conda_env=""
    local script=""
    case "$model" in
        alphagenome)
            conda_env="alphagenome"
            script="genperturb/preprocess/20_alphagenome_embedding.py"
            ;;
        alphagenome_prediction)
            conda_env="alphagenome"
            script="genperturb/preprocess/20a_alphagenome_prediction.py"
            ;;
        enformer)
            conda_env="enformer"
            script="genperturb/preprocess/21_enformer_embedding.py"
            ;;
        borzoi)
            conda_env="borzoi"
            script="genperturb/preprocess/22_borzoi_embedding.py"
            ;;
        *)
            echo "Unknown model: $model" >&2
            return 1
            ;;
    esac

    sbatch \
        -J "embed_${model}" \
        -o "log/00a_embed_${model}.out" \
        -e "log/00a_embed_${model}.err" \
        --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
        --ntasks=1 --cpus-per-task=1 \
        <<SCRIPT
#!/bin/bash
source ${CONDA_SH}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="\${TMPDIR:-/tmp}/genperturb_numba_\${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="\${TMPDIR:-/tmp}/genperturb_mpl_\${SLURM_JOB_ID:-manual}"
mkdir -p "\${NUMBA_CACHE_DIR}" "\${MPLCONFIGDIR}"
conda activate ${conda_env}
srun python ${script}
SCRIPT
}

echo "=== 00a: Submitting embedding extraction (${MODELS[*]}) ==="
for model in "${MODELS[@]}"; do
    submit_embedding "$model"
done

echo "Submit alphagenome reference predictions (needed by Fig 2 baselines):"
echo "  bash $0 alphagenome_prediction"
