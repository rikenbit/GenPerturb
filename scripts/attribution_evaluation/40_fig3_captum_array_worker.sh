#!/bin/bash
#
# SLURM array worker for paper Fig. 3 root-output Captum routes.
#
# Supported task routes:
#   fig3d_gtgenes  -> 10_captum.py union_genes ... --output-suffix _gtgenes
#   fig3e_top200   -> 10_captum.py variable_genes condition
#
# This worker writes raw H5 files under attribution/{study}__{suffix}/... .
# It does not use attribution_analysis/ and should not be used for Fig. 4.

#SBATCH --job-name=fig3_captum
#SBATCH --partition=h200-long
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=log/fig3_captum_%A_%a.out
#SBATCH --error=log/fig3_captum_%A_%a.err
#SBATCH --nice=100

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
TASKS_FILE="${TASKS_FILE:?TASKS_FILE is required}"
DONE_DIR="${DONE_DIR:?DONE_DIR is required}"
CONDA_SH="${CONDA_SH:?CONDA_SH is required}"

cd "$PROJECT_ROOT"

TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
if [ -z "$TASK_LINE" ]; then
    echo "ERROR: No task found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r TASK_ID ROUTE STUDY STUDY_SUFFIX MODEL FIELD5 FIELD6 FIELD7 FIELD8 REST <<< "$TASK_LINE"
STUDY_FULL="${STUDY}__${STUDY_SUFFIX}"

mkdir -p "$DONE_DIR"
DONE_FLAG="${DONE_DIR}/${TASK_ID}.ok"
if [ -f "$DONE_FLAG" ]; then
    echo "SKIP: ${DONE_FLAG} exists."
    exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
set +u
source "$CONDA_SH"
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

echo "========================================"
echo "Fig. 3 Captum task ${TASK_ID}"
echo "Route:        ${ROUTE}"
echo "Study:        ${STUDY_FULL}"
echo "Model:        ${MODEL}"
echo "Node:         $(hostname)"
echo "Start time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

if [ "$ROUTE" == "fig3d_gtgenes" ]; then
    TARGET="$FIELD5"
    PERT="$FIELD6"
    GENE_LIST="$FIELD7"
    OUTPUT_SUFFIX="$FIELD8"
    SAFE_PERT="${PERT//\//_}"
    RAW_H5="attribution/${STUDY_FULL}/${SAFE_PERT}/${SAFE_PERT}${OUTPUT_SUFFIX}_raw_attribution.h5"

    echo "Fig. 3d enhancer AUPRC route: Martin Table S3 genes"
    echo "Pert:         ${PERT}"
    echo "Target:       ${TARGET}"
    echo "Gene list:    ${GENE_LIST}"
    echo "Raw output:   ${RAW_H5}"

    if [ -s "$RAW_H5" ]; then
        echo "SKIP: raw H5 already exists: ${RAW_H5}"
    else
        python scripts/attribution_evaluation/10_captum.py \
            "$STUDY" "$STUDY_SUFFIX" union_genes "$TARGET" \
            --pert "$PERT" \
            --gene-list "$GENE_LIST" \
            --output-suffix "$OUTPUT_SUFFIX"
    fi
    OUTPUT_PATH="$RAW_H5"

elif [ "$ROUTE" == "fig3e_top200" ]; then
    MODE="$FIELD5"
    TARGET="$FIELD6"
    PERT="$FIELD7"
    SAFE_PERT="${PERT//\//_}"
    RAW_H5="attribution/${STUDY_FULL}/${SAFE_PERT}/${SAFE_PERT}_raw_attribution.h5"

    echo "Fig. 3e motif route: top-200 variable genes"
    echo "Pert:         ${PERT}"
    echo "Mode/target:  ${MODE}/${TARGET}"
    echo "Raw output:   ${RAW_H5}"

    if [ -s "$RAW_H5" ]; then
        echo "SKIP: raw H5 already exists: ${RAW_H5}"
    else
        python scripts/attribution_evaluation/10_captum.py \
            "$STUDY" "$STUDY_SUFFIX" "$MODE" "$TARGET" \
            --pert "$PERT"
    fi
    OUTPUT_PATH="$RAW_H5"

else
    echo "ERROR: Unknown route '${ROUTE}' in ${TASKS_FILE}" >&2
    exit 1
fi

if [ ! -s "$OUTPUT_PATH" ]; then
    echo "ERROR: expected raw H5 was not created: ${OUTPUT_PATH}" >&2
    exit 1
fi

cat > "$DONE_FLAG" <<EOF
status=success
task_id=$TASK_ID
route=$ROUTE
study=$STUDY
study_suffix=$STUDY_SUFFIX
model=$MODEL
output=$OUTPUT_PATH
node=$(hostname)
slurm_job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
completed=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo "Done flag: ${DONE_FLAG}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
