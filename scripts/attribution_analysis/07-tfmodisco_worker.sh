#!/bin/bash
#
# Reads task from tasks.txt, runs TF-MoDISco on the peak-window
# attribution H5 and BED files from the captum+peak pipeline.
#
# Prerequisites:
#   - Captum+peak pipeline done flag exists for this task
#   - Peak H5 and BED files in attribution_analysis/attribution/{study_full}/{pert}/
#
# Required environment variables (set by submit script):
#   PROJECT_ROOT, TASKS_FILE, OUTPUT_BASE, CAPTUM_DONE_DIR, CONDA_SH
#

#SBATCH --job-name=tfmodisco
#SBATCH --partition=cpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=log/tfmodisco_%A_%a.out
#SBATCH --error=log/tfmodisco_%A_%a.err
#SBATCH --nice=100

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
TASKS_FILE="${TASKS_FILE:-attribution_analysis/tasks.txt}"
DONE_DIR="${DONE_DIR:-attribution_analysis/done_modisco}"
OUTPUT_BASE="${OUTPUT_BASE:-attribution_analysis}"
CAPTUM_DONE_DIR="${CAPTUM_DONE_DIR:-attribution_analysis/done}"
INPUT_BASE="${INPUT_BASE:-attribution_analysis}"
MODISCO_CONDITION="${MODISCO_CONDITION:-short}"

cd "$PROJECT_ROOT"

TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
if [ -z "$TASK_LINE" ]; then
    echo "ERROR: No task for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

TASK_ID=$(echo "$TASK_LINE" | cut -f1)
STUDY=$(echo "$TASK_LINE" | cut -f2)
STUDY_SUFFIX=$(echo "$TASK_LINE" | cut -f3)
MODEL=$(echo "$TASK_LINE" | cut -f4)
PERT=$(echo "$TASK_LINE" | cut -f5)
SAFE_PERT="${PERT//\//_}"

echo "========================================"
echo "TF-MoDISco Task ${TASK_ID}: ${PERT}"
echo "========================================"
echo "Node: $(hostname), Start: $(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$DONE_DIR"
DONE_FLAG="${DONE_DIR}/${TASK_ID}.ok"

if [ -f "$DONE_FLAG" ]; then
    echo "SKIP: Already completed (${DONE_FLAG})"
    exit 0
fi

CAPTUM_FLAG="${CAPTUM_DONE_DIR}/${TASK_ID}.ok"
if [ ! -f "$CAPTUM_FLAG" ]; then
    echo "SKIP: Captum+peak not yet done for task ${TASK_ID}." >&2
    echo "Expected: ${CAPTUM_FLAG}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

set +u
source "${CONDA_SH:?CONDA_SH is required. Submit via pipeline/fig4_lineage/22_tfmodisco_array.sh or export CONDA_SH.}"
conda activate modisco
set -u

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMBA_NUM_THREADS=16

echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

STEP_RC=0
python scripts/attribution_analysis/05-run_tfmodisco.py \
    --study "$STUDY" \
    --study-suffix "$STUDY_SUFFIX" \
    --pert "$PERT" \
    --input-base "$INPUT_BASE" \
    --output-base "$OUTPUT_BASE" \
    --condition "$MODISCO_CONDITION" || STEP_RC=$?

if [ $STEP_RC -ne 0 ]; then
    echo "ERROR: TF-MoDISco failed with exit code $STEP_RC" >&2
    exit $STEP_RC
fi

cat > "$DONE_FLAG" <<EOF
status=success
task_id=$TASK_ID
pert=$PERT
study=$STUDY
node=$(hostname)
slurm_job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
completed=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo "Done: ${DONE_FLAG}"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
