#!/bin/bash
# ==============================================================================
# Fig 4 / 10: Captum + peak-call array for attribution_analysis route (S11-S12)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-NormanWeissman2019_filtered_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
shift $(( $# >= 2 ? 2 : $# ))

SUFFIX_OVERRIDE=""
NODELIST=""
MODE="variable_genes"
TARGET="all"
PARTITION="${PARTITION_GPU}"
GRES="${GPU_GRES}"
MEM="${MEM_GPU}"
TIME="2:00:00"
MAX_CONCURRENT=16

while [ $# -gt 0 ]; do
    case "$1" in
        --suffix)         SUFFIX_OVERRIDE="$2"; shift 2 ;;
        --nodelist)       NODELIST="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;
        --target)         TARGET="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --gres)           GRES="$2"; shift 2 ;;
        --mem)            MEM="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

STUDY_SUFFIX="${SUFFIX_OVERRIDE:-$(default_study_suffix "$MODEL")}"
STUDY_FULL="${STUDY}__${STUDY_SUFFIX}"

TASKS_DIR="attribution_analysis"
TASKS_FILE="${TASKS_DIR}/tasks_${STUDY_FULL}.txt"
DONE_DIR="${TASKS_DIR}/done/${STUDY_FULL}"
mkdir -p "$TASKS_DIR" "$DONE_DIR"

echo "Fig 4 attribution_analysis Captum + Peak Array — study=${STUDY_FULL} mode=${MODE} target=${TARGET}"

# --- 1. generate task list ---------------------------------------------------
pipeline_activate_conda alphagenome
GENERATE_ARGS=("$STUDY" "$MODEL" --output "$TASKS_FILE")
[ -n "$SUFFIX_OVERRIDE" ] && GENERATE_ARGS+=(--suffix "$SUFFIX_OVERRIDE")
python scripts/attribution_analysis/01-generate_tasks.py "${GENERATE_ARGS[@]}"

# --- 2. submit resume-safe array ---------------------------------------------
PENDING_IDS=()
while IFS=$'\t' read -r TASK_ID REST; do
    [ -f "${DONE_DIR}/${TASK_ID}.ok" ] || PENDING_IDS+=("$TASK_ID")
done < "$TASKS_FILE"

echo "Tasks: $(wc -l < "$TASKS_FILE")  To submit: ${#PENDING_IDS[@]}"
[ ${#PENDING_IDS[@]} -eq 0 ] && { echo "Nothing to submit."; exit 0; }

ARRAY_SPEC=$(IFS=,; echo "${PENDING_IDS[*]}")
ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"

SBATCH_ARGS=(
    --parsable --array="$ARRAY_SPEC"
    --partition="$PARTITION" --gres="${GRES}" --mem="$MEM" --time="$TIME"
    --cpus-per-task=4 --nice=100
    --job-name="fig4_10_captum_${STUDY%%_*}"
    --output="log/fig4_10_captum_%A_%a.out"
    --error="log/fig4_10_captum_%A_%a.err"
)
[ -n "$NODELIST" ] && SBATCH_ARGS+=(--nodelist="$NODELIST")

export PROJECT_ROOT TASKS_FILE DONE_DIR CONDA_SH
export OUTPUT_BASE="${TASKS_DIR}"
export CAPTUM_MODE="$MODE"
export CAPTUM_TARGET="$TARGET"

JOBID=$(sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,PROJECT_ROOT,TASKS_FILE,DONE_DIR,OUTPUT_BASE,CAPTUM_MODE,CAPTUM_TARGET,CONDA_SH \
    scripts/attribution_analysis/02-array_worker.sh)
echo "Submitted Fig 4 attribution_analysis job array: $JOBID  (${#PENDING_IDS[@]} tasks)"
