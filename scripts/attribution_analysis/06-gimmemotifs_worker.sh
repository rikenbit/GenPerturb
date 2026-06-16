#!/bin/bash
#
# Reads task from tasks.txt, runs gimme motifs on the peak BED file
# produced by the captum+peak pipeline.
#
# Prerequisites:
#   - Captum+peak pipeline done flag exists for this task
#   - Peak BED files in attribution_analysis/attribution/{study_full}/{pert}/
#
# Required environment variables (set by submit script):
#   PROJECT_ROOT, TASKS_FILE, OUTPUT_BASE, CAPTUM_DONE_DIR, CONDA_SH
#

#SBATCH --job-name=gimmemotifs
#SBATCH --partition=h200-long
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=log/gimmemotifs_%A_%a.out
#SBATCH --error=log/gimmemotifs_%A_%a.err
#SBATCH --nice=100

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
TASKS_FILE="${TASKS_FILE:-attribution_analysis/tasks.txt}"
DONE_DIR="${DONE_DIR:-attribution_analysis/done_gimme}"
OUTPUT_BASE="${OUTPUT_BASE:-attribution_analysis}"
CAPTUM_DONE_DIR="${CAPTUM_DONE_DIR:-attribution_analysis/done}"
INPUT_BASE="${INPUT_BASE:-attribution_analysis}"
THREADS="${GIMME_THREADS:-16}"
BED_TYPE="${BED_TYPE:-attribution}"

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
echo "GimmeMotifs Task ${TASK_ID}: ${PERT}"
echo "========================================"
echo "Node: $(hostname), Start: $(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$DONE_DIR"
DONE_FLAG="${DONE_DIR}/${TASK_ID}.ok"

if [ -f "$DONE_FLAG" ]; then
    echo "SKIP: Already completed (${DONE_FLAG})"
    exit 0
fi

if [ "$BED_TYPE" = "attribution" ]; then
    # Attribution mode: captum+peak must be done
    CAPTUM_FLAG="${CAPTUM_DONE_DIR}/${TASK_ID}.ok"
    if [ ! -f "$CAPTUM_FLAG" ]; then
        echo "SKIP: Captum+peak not yet done for task ${TASK_ID}. Waiting." >&2
        echo "Expected: ${CAPTUM_FLAG}" >&2
        exit 1
    fi
else
    # CRE mode: check CRE BED file existence
    STUDY_FULL="${STUDY}__${STUDY_SUFFIX}"
    CRE_BED="${INPUT_BASE}/cre/${STUDY_FULL}/${SAFE_PERT}/${BED_TYPE}_${SAFE_PERT}.bed"
    if [ ! -f "$CRE_BED" ]; then
        echo "SKIP: CRE BED not found: $CRE_BED (task ${TASK_ID})"
        exit 0
    fi
    echo "CRE BED found: $CRE_BED"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Isolate genomepy cache per job to avoid TOCTOU race on shared Lustre filesystem.
# Without this, concurrent array jobs all share ~/.cache/genomepy/*/cache.lock
# and one job can delete the lock between another job's exists() and stat() calls.
export XDG_CACHE_HOME="/tmp/genomepy_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$XDG_CACHE_HOME"

set +u
source "${CONDA_SH:?CONDA_SH is required. Submit via pipeline/fig4_lineage/20_gimmemotifs_array.sh or export CONDA_SH.}"
conda activate gimme
set -u

echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

if ! command -v gimme &>/dev/null; then
    echo "ERROR: gimme not found in PATH" >&2
    exit 1
fi

STEP_RC=0
python scripts/attribution_analysis/04-run_gimmemotifs.py \
    --study "$STUDY" \
    --study-suffix "$STUDY_SUFFIX" \
    --pert "$PERT" \
    --input-base "$INPUT_BASE" \
    --output-base "$OUTPUT_BASE" \
    --bed-type "$BED_TYPE" \
    --threads "$THREADS" || STEP_RC=$?

if [ $STEP_RC -ne 0 ]; then
    echo "ERROR: GimmeMotifs failed with exit code $STEP_RC" >&2
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
