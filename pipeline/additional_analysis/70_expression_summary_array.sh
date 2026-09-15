#!/bin/bash
# ==============================================================================
# 70: Saved-prediction expression summaries, 56 configurations
#   Fig. 3 backbone comparison and Fig. S5 fold comparison numbers
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

OUTDIR="${OUTDIR:-results/expression_evaluation}"
INDICES="${INDICES:-0-55}"

sbatch --parsable \
    -J "step70_expression" \
    -a "${INDICES}" \
    -o "log/step70_expression_%a.out" \
    -e "log/step70_expression_%a.err" \
    --partition="${PARTITION_CPU}" --mem="${MEM_HEAVY}" --time=24:00:00 \
    <<SCRIPT
#!/bin/bash
source ${CONDA_SH}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
conda activate singlecell
python scripts/additional_analysis/fig3_figS5_expression_batch.py \
    --root "${PROJECT_ROOT}" --out "${PROJECT_ROOT}/${OUTDIR}" --index \${SLURM_ARRAY_TASK_ID}
SCRIPT
