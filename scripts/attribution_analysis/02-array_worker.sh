#!/bin/bash
#
# This script is submitted as a SLURM job array. Each array task processes
# one perturbation:
#   1. Run 10_captum.py (raw attribution to node-local TMPDIR)
#   2. Run 10a_peak_call.py (peak calling on raw attribution)
#   3. Copy peak results to permanent storage (attribution_analysis/)
#   4. TMPDIR auto-cleaned (no raw .h5 on shared storage)
#
# Idempotency: tasks with existing done/<task_id>.ok are skipped.
#
# Required environment variables (set by submit script):
#   PROJECT_ROOT  - absolute path to GenPerturb repo root
#   TASKS_FILE    - path to tasks.txt (relative to PROJECT_ROOT)
#   DONE_DIR      - path to done flag directory (relative to PROJECT_ROOT)
#   OUTPUT_BASE   - path to output base directory (relative to PROJECT_ROOT)
#   CAPTUM_MODE   - analysis mode (variable_genes, across_genes, across_perturbations)
#   CAPTUM_TARGET - target selection (all, tf, condition, etc.)
#   CONDA_SH      - path to conda profile script (provided by pipeline/_common.sh)
#
# Usage (via sbatch, not directly):
#   sbatch --array=0-N%8 scripts/attribution_analysis/02-array_worker.sh
#

#SBATCH --job-name=captum_peak
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=log/captum_peak_%A_%a.out
#SBATCH --error=log/captum_peak_%A_%a.err
#SBATCH --nice=100

# --- Strict error handling ---
# NOTE: We do NOT use "set -e" globally because we need to capture exit codes
# from individual steps for the done-flag logic. Instead, we check $? explicitly.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
TASKS_FILE="${TASKS_FILE:-attribution_analysis/tasks.txt}"
DONE_DIR="${DONE_DIR:-attribution_analysis/done}"
OUTPUT_BASE="${OUTPUT_BASE:-attribution_analysis}"
MODE="${CAPTUM_MODE:-variable_genes}"
TARGET="${CAPTUM_TARGET:-all}"

cd "$PROJECT_ROOT"

TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
if [ -z "$TASK_LINE" ]; then
    echo "ERROR: No task found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

TASK_ID=$(echo "$TASK_LINE" | cut -f1)
STUDY=$(echo "$TASK_LINE" | cut -f2)
STUDY_SUFFIX=$(echo "$TASK_LINE" | cut -f3)
MODEL=$(echo "$TASK_LINE" | cut -f4)
PERT=$(echo "$TASK_LINE" | cut -f5)
SAFE_PERT="${PERT//\//_}"
STUDY_FULL="${STUDY}__${STUDY_SUFFIX}"

echo "========================================"
echo "Task ${TASK_ID}: ${PERT}"
echo "========================================"
echo "Study:        $STUDY"
echo "Study suffix: $STUDY_SUFFIX"
echo "Model:        $MODEL"
echo "Mode:         $MODE"
echo "Target:       $TARGET"
echo "SLURM Job:    ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

mkdir -p "$DONE_DIR"
DONE_FLAG="${DONE_DIR}/${TASK_ID}.ok"

if [ -f "$DONE_FLAG" ]; then
    echo "SKIP: ${DONE_FLAG} exists. Task already completed."
    exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

set +u
source "${CONDA_SH:?CONDA_SH is required. Submit via pipeline/fig4_lineage/10_captum_array.sh or export CONDA_SH.}"
set -u

if [ "$MODEL" == "enformer" ] || [ "$MODEL" == "enformer_masked" ]; then
    conda activate enformer
elif [ "$MODEL" == "borzoi" ]; then
    conda activate borzoi
elif [[ "$MODEL" == alphagenome* ]]; then
    conda activate alphagenome
elif [ "$MODEL" == "simplecnn" ]; then
    conda activate bend
else
    echo "WARNING: Unknown model '$MODEL', not activating a specific conda env."
fi

echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

# Verify macs3 is available (required by 10a_peak_call.py)
if ! command -v macs3 &>/dev/null; then
    echo "WARNING: macs3 not found in PATH. 10a_peak_call.py will fail." >&2
    echo "Install via: pip install macs3  (in the active conda env)" >&2
fi

# SLURM_TMPDIR: node-local SSD/RAM provided by SLURM (auto-cleaned on job end).
# Fallback: /tmp with unique name (manually cleaned on exit).
if [ -n "${SLURM_TMPDIR:-}" ]; then
    TMPWORK="$SLURM_TMPDIR/captum_work_${SLURM_ARRAY_TASK_ID}"
    CREATED_TMPWORK=false
else
    TMPWORK="/tmp/captum_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$$"
    CREATED_TMPWORK=true
fi
mkdir -p "$TMPWORK"
echo "Temp workspace: $TMPWORK"

# Cleanup function: remove tmpdir contents on exit (belt-and-suspenders)
cleanup() {
    local exit_code=$?
    echo ""
    echo "Cleaning up temp workspace..."
    rm -rf "$TMPWORK/attribution" "$TMPWORK/attribution_pert" "$TMPWORK/attribution_seq" 2>/dev/null || true
    if $CREATED_TMPWORK; then
        rm -rf "$TMPWORK" 2>/dev/null || true
    fi
    echo "Cleanup done. Exit code: $exit_code"
}
trap cleanup EXIT

# Check TMPDIR has sufficient space (~5GB needed per task for alphagenome)
TMPDIR_AVAIL_KB=$(df --output=avail "$TMPWORK" 2>/dev/null | tail -1 | tr -d ' ')
if [ -n "$TMPDIR_AVAIL_KB" ] && [ "$TMPDIR_AVAIL_KB" -lt 5242880 ]; then
    echo "WARNING: TMPDIR has only ${TMPDIR_AVAIL_KB}KB free (<5GB). May be insufficient." >&2
fi

echo ""
echo "========== Step 1: Captum Attribution =========="
echo "Output base: $TMPWORK"

STEP1_RC=0
python scripts/attribution_evaluation/10_captum.py \
    "$STUDY" "$STUDY_SUFFIX" "$MODE" "$TARGET" \
    --pert "$PERT" \
    --base-dir "$TMPWORK" || STEP1_RC=$?

if [ $STEP1_RC -ne 0 ]; then
    echo "ERROR: 10_captum.py failed with exit code $STEP1_RC" >&2
    echo "Pert: $PERT, Task: $TASK_ID" >&2
    exit $STEP1_RC
fi

# Verify raw h5 was created
RAW_H5_COUNT=$(find "$TMPWORK" -name "*_raw_attribution.h5" 2>/dev/null | wc -l)
echo "Raw h5 files in TMPDIR: $RAW_H5_COUNT"
if [ "$RAW_H5_COUNT" -eq 0 ]; then
    echo "ERROR: No raw attribution h5 found in TMPDIR after captum step" >&2
    exit 1
fi

echo ""
echo "========== Step 2: Peak Calling =========="

STEP2_RC=0
python scripts/attribution_evaluation/10a_peak_call.py \
    "$STUDY" "$STUDY_SUFFIX" "$MODE" \
    --base-dir "$TMPWORK" || STEP2_RC=$?

if [ $STEP2_RC -ne 0 ]; then
    echo "ERROR: 10a_peak_call.py failed with exit code $STEP2_RC" >&2
    echo "Pert: $PERT, Task: $TASK_ID" >&2
    exit $STEP2_RC
fi

echo ""
echo "========== Step 3: Copy Results =========="

# Determine source and destination based on mode
if [ "$MODE" == "variable_genes" ]; then
    SRC_DIR="$TMPWORK/attribution/$STUDY_FULL/$SAFE_PERT"
    DST_DIR="$OUTPUT_BASE/attribution/$STUDY_FULL/$SAFE_PERT"
elif [ "$MODE" == "across_genes" ]; then
    SRC_DIR="$TMPWORK/attribution_pert/$STUDY_FULL/$SAFE_PERT"
    DST_DIR="$OUTPUT_BASE/attribution_pert/$STUDY_FULL/$SAFE_PERT"
else
    SRC_DIR="$TMPWORK/attribution_seq/$STUDY_FULL"
    DST_DIR="$OUTPUT_BASE/attribution_seq/$STUDY_FULL"
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found: $SRC_DIR" >&2
    echo "Listing TMPWORK contents:" >&2
    find "$TMPWORK" -type f 2>/dev/null | head -20 >&2
    exit 1
fi

mkdir -p "$DST_DIR"

# Copy everything EXCEPT *_raw_attribution.h5 (peak files only)
find "$SRC_DIR" -type f ! -name "*_raw_attribution.h5" -exec cp -v {} "$DST_DIR/" \;

echo ""
echo "Copied peak results to: $DST_DIR"
echo "Files in output:"
ls -lh "$DST_DIR/" 2>/dev/null || true

# Safety check: no raw h5 in permanent storage
RAW_IN_DST=$(find "$DST_DIR" -name "*_raw_attribution.h5" 2>/dev/null | wc -l)
if [ "$RAW_IN_DST" -gt 0 ]; then
    echo "WARNING: Raw h5 found in output directory! Removing..."
    find "$DST_DIR" -name "*_raw_attribution.h5" -delete
fi

echo ""
echo "========== Task Complete =========="
cat > "$DONE_FLAG" <<EOF
status=success
task_id=$TASK_ID
pert=$PERT
study=$STUDY
study_suffix=$STUDY_SUFFIX
model=$MODEL
mode=$MODE
node=$(hostname)
slurm_job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
completed=$(date '+%Y-%m-%d %H:%M:%S')
output_dir=$DST_DIR
EOF

echo "Done flag created: $DONE_FLAG"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
