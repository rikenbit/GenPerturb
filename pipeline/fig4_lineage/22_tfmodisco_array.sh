#!/bin/bash
# ==============================================================================
# Fig 4 / 22: TF-MoDISco array (CPU array, study-parameterized)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-NormanWeissman2019_filtered_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
shift $(( $# >= 2 ? 2 : $# ))

SUFFIX_OVERRIDE=""
TASKS_FILE=""
NODELIST=""
MAX_CONCURRENT=10
PARTITION="${PARTITION_CPU}"
MEM="120G"
TIME="12:00:00"
CONDITION="short"

while [ $# -gt 0 ]; do
    case "$1" in
        --suffix)         SUFFIX_OVERRIDE="$2"; shift 2 ;;
        --tasks-file)     TASKS_FILE="$2"; shift 2 ;;
        --nodelist)       NODELIST="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --mem)            MEM="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --condition)      CONDITION="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

STUDY_SUFFIX="${SUFFIX_OVERRIDE:-$(default_study_suffix "$MODEL")}"
STUDY_FULL="${STUDY}__${STUDY_SUFFIX}"

[ -z "$TASKS_FILE" ] && TASKS_FILE="attribution_analysis/tasks_${STUDY_FULL}.txt"
DONE_DIR="attribution_analysis/done_modisco/${STUDY_FULL}"
CAPTUM_DONE_DIR="attribution_analysis/done/${STUDY_FULL}"
mkdir -p "$DONE_DIR"

PENDING_IDS=()
while IFS=$'\t' read -r TASK_ID REST; do
    [ -f "${CAPTUM_DONE_DIR}/${TASK_ID}.ok" ] || continue
    [ -f "${DONE_DIR}/${TASK_ID}.ok" ] && continue
    PENDING_IDS+=("$TASK_ID")
done < "$TASKS_FILE"

echo "Captum done: $(find "$CAPTUM_DONE_DIR" -name '*.ok' 2>/dev/null | wc -l)  TF-MoDISco to submit: ${#PENDING_IDS[@]}"
[ ${#PENDING_IDS[@]} -eq 0 ] && { echo "Nothing to submit."; exit 0; }

ARRAY_SPEC=$(IFS=,; echo "${PENDING_IDS[*]}")
ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"

SBATCH_ARGS=(
    --parsable --array="$ARRAY_SPEC"
    --partition="$PARTITION" --mem="$MEM" --time="$TIME"
    --cpus-per-task=16 --nice=100
    --job-name="fig4_22_modisco"
    --output="log/fig4_22_modisco_%A_%a.out"
    --error="log/fig4_22_modisco_%A_%a.err"
)
[ -n "$NODELIST" ] && SBATCH_ARGS+=(--nodelist="$NODELIST")

export PROJECT_ROOT TASKS_FILE DONE_DIR CAPTUM_DONE_DIR CONDA_SH
export OUTPUT_BASE="attribution_analysis"
export INPUT_BASE="attribution_analysis"
export MODISCO_CONDITION="$CONDITION"

JOBID=$(sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,PROJECT_ROOT,TASKS_FILE,DONE_DIR,OUTPUT_BASE,INPUT_BASE,CAPTUM_DONE_DIR,MODISCO_CONDITION,CONDA_SH \
    scripts/attribution_analysis/07-tfmodisco_worker.sh)
echo "Submitted: $JOBID (${#PENDING_IDS[@]} tasks)"
