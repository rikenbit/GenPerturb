#!/bin/bash
# ==============================================================================
# Fig 3d / S9: Captum for Martin enhancer AUPRC gtgenes route
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
shift $(( $# >= 2 ? 2 : $# ))

SUFFIX_OVERRIDE=""
TARGET="condition"
PERTS=""
NODELIST=""
PARTITION="${PARTITION_GPU}"
GRES="${GPU_GRES}"
MEM="${MEM_GPU}"
TIME="8:00:00"
MAX_CONCURRENT=4

while [ $# -gt 0 ]; do
    case "$1" in
        --suffix)         SUFFIX_OVERRIDE="$2"; shift 2 ;;
        --target)         TARGET="$2"; shift 2 ;;
        --perts)          PERTS="$2"; shift 2 ;;
        --nodelist)       NODELIST="$2"; shift 2 ;;
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

TASKS_DIR="attribution/fig3d_gtgenes_tasks"
TASKS_FILE="${TASKS_DIR}/tasks_${STUDY_FULL}.txt"
DONE_DIR="${TASKS_DIR}/done/${STUDY_FULL}"
mkdir -p "$TASKS_DIR" "$DONE_DIR"

echo "Fig 3d / S9 Captum gtgenes — study=${STUDY_FULL} target=${TARGET}"

# --- 1. generate task list ---------------------------------------------------
pipeline_activate_conda alphagenome
GENERATE_ARGS=(fig3d_gtgenes "$STUDY" "$MODEL" --target "$TARGET" --output "$TASKS_FILE")
[ -n "$SUFFIX_OVERRIDE" ] && GENERATE_ARGS+=(--suffix "$SUFFIX_OVERRIDE")
[ -n "$PERTS" ] && GENERATE_ARGS+=(--perts "$PERTS")
python scripts/attribution_evaluation/40_generate_fig3_captum_tasks.py "${GENERATE_ARGS[@]}"

# --- 2. submit resume-safe Captum array --------------------------------------
PENDING_IDS=()
while IFS=$'\t' read -r TASK_ID ROUTE STUDY_FIELD SUFFIX_FIELD MODEL_FIELD TARGET_FIELD PERT_FIELD GENE_LIST_FIELD OUTPUT_SUFFIX_FIELD; do
    SAFE_PERT="${PERT_FIELD//\//_}"
    RAW_H5="attribution/${STUDY_FIELD}__${SUFFIX_FIELD}/${SAFE_PERT}/${SAFE_PERT}${OUTPUT_SUFFIX_FIELD}_raw_attribution.h5"
    [ -f "${DONE_DIR}/${TASK_ID}.ok" ] || [ -s "$RAW_H5" ] || PENDING_IDS+=("$TASK_ID")
done < "$TASKS_FILE"

echo "Tasks: $(wc -l < "$TASKS_FILE")  To submit: ${#PENDING_IDS[@]}"

if [ "${#PENDING_IDS[@]}" -eq 0 ]; then
    echo "All Fig 3d / S9 gtgenes Captum raw H5 files already exist; no SLURM jobs submitted."
    echo "Peak calling is not run automatically in this no-pending case to avoid overwriting existing outputs."
    exit 0
fi

ARRAY_SPEC=$(IFS=,; echo "${PENDING_IDS[*]}")
ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"

SBATCH_ARGS=(
    --parsable --array="$ARRAY_SPEC"
    --partition="$PARTITION" --gres="${GRES}" --mem="$MEM" --time="$TIME"
    --cpus-per-task=4 --nice=100
    --job-name="fig3d_gtgenes_${STUDY%%_*}"
    --output="log/fig3d_gtgenes_%A_%a.out"
    --error="log/fig3d_gtgenes_%A_%a.err"
)
[ -n "$NODELIST" ] && SBATCH_ARGS+=(--nodelist="$NODELIST")

export PROJECT_ROOT TASKS_FILE DONE_DIR CONDA_SH
ARRAY_JOBID=$(sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,PROJECT_ROOT,TASKS_FILE,DONE_DIR,CONDA_SH \
    scripts/attribution_evaluation/40_fig3_captum_array_worker.sh)
echo "Submitted Fig 3d gtgenes Captum array: $ARRAY_JOBID"

# --- 3. submit dependent peak-call job ---------------------------------------
PEAK_JOBID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOBID}" \
    --partition="$PARTITION_CPU" --mem="$MEM_LIGHT" --time="4:00:00" \
    --cpus-per-task=4 --nice=100 \
    --job-name="fig3d_gtgenes_peak_${STUDY%%_*}" \
    --output="log/fig3d_gtgenes_peak_%j.out" \
    --error="log/fig3d_gtgenes_peak_%j.err" \
    --wrap="bash pipeline/fig3_attribution/12_peak_call.sh '$STUDY' '$MODEL' '$STUDY_SUFFIX' _gtgenes")
echo "Submitted dependent peak-call job: $PEAK_JOBID"
