#!/bin/bash
# ==============================================================================
# Fig 3f / 42: In-silico mutation GPU array
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-Martin_full}"

CWD="${PROJECT_ROOT}"

STUDY_DIR="attribution_analysis/insilico_mutation/${STUDY}"
LOGDIR="${STUDY_DIR}/log"
TASKFILE="${STUDY_DIR}/tasks.txt"
RESULTDIR="${STUDY_DIR}/results"
PARTITION="h200-long"
MAX_PER_NODE=8
MEM_PER_JOB="60G"

mkdir -p "$LOGDIR"
mkdir -p "$RESULTDIR"

NTASKS=$(wc -l < "$TASKFILE")
echo "Study:       $STUDY"
echo "Total tasks: $NTASKS genes"
echo "Target:      gpu02 (max $MAX_PER_NODE) + gpu03 (max $MAX_PER_NODE) = 16 parallel"
echo

TASKFILE_02="${STUDY_DIR}/tasks_gpu02.txt"
TASKFILE_03="${STUDY_DIR}/tasks_gpu03.txt"

awk 'NR%2==1' "$TASKFILE" > "$TASKFILE_02"
awk 'NR%2==0' "$TASKFILE" > "$TASKFILE_03"

N02=$(wc -l < "$TASKFILE_02")
N03=$(wc -l < "$TASKFILE_03")
echo "  -> gpu02 partition: $N02 tasks ($TASKFILE_02)"
echo "  -> gpu03 partition: $N03 tasks ($TASKFILE_03)"
echo

submit_node() {
    local NODE="$1"
    local NODE_TASKFILE="$2"
    local NODE_NTASKS="$3"

    local MAX_ARRAY=1000
    local OFFSET=0

    while [ $OFFSET -lt $NODE_NTASKS ]; do
        local BATCH_SIZE=$((NODE_NTASKS - OFFSET))
        if [ $BATCH_SIZE -gt $MAX_ARRAY ]; then
            BATCH_SIZE=$MAX_ARRAY
        fi

        echo "  Submitting on $NODE: offset=${OFFSET}, size=${BATCH_SIZE}"

        local JOBID
        JOBID=$(sbatch --parsable \
            --job-name="mut52_${STUDY}_${NODE}" \
            --partition="$PARTITION" \
            --nodelist="$NODE" \
            --nodes=1 \
            --gres=gpu:h200:1 \
            --mem="$MEM_PER_JOB" \
            --cpus-per-task=2 \
            --array=1-${BATCH_SIZE}%${MAX_PER_NODE} \
            --output="${LOGDIR}/mutation_${NODE}_%A_%a.out" \
            --error="${LOGDIR}/mutation_${NODE}_%A_%a.err" \
            --export=ALL,TASK_OFFSET=${OFFSET},STUDY=${STUDY},NODE_TASKFILE=${NODE_TASKFILE},PROJECT_ROOT=${CWD},CONDA_SH=${CONDA_SH} \
            <<'SBATCH_EOF'
#!/bin/bash
set -euo pipefail

CWD="${PROJECT_ROOT:?PROJECT_ROOT is required}"
cd "$CWD"
export PYTHONPATH="$CWD:${PYTHONPATH:-}"

set +u
source "${CONDA_SH:?CONDA_SH is required}"
conda activate alphagenome
set -u

STUDY_DIR="attribution_analysis/insilico_mutation/${STUDY}"
LINE_NUM=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
GENE=$(sed -n "${LINE_NUM}p" "$NODE_TASKFILE")

if [ -z "$GENE" ]; then
    echo "No gene found at line $LINE_NUM in $NODE_TASKFILE — skipping"
    exit 0
fi

RESULTFILE="${STUDY_DIR}/results/${GENE}_predictions.h5"
if [ -f "$RESULTFILE" ]; then
    echo "Already computed: $GENE — skipping"
    exit 0
fi

echo "Task line=${LINE_NUM} (taskfile=${NODE_TASKFILE}): gene=$GENE, study=$STUDY"
echo "Node: $(hostname)"
echo "Start: $(date)"

python scripts/attribution_evaluation/52b_insilico_mutation.py \
    --study "$STUDY" \
    --gene "$GENE" \
    --n-seeds 5

echo "End: $(date)"
SBATCH_EOF
)
        echo "    Submitted $JOBID (${BATCH_SIZE} tasks, max ${MAX_PER_NODE} parallel on $NODE)"
        OFFSET=$((OFFSET + BATCH_SIZE))
    done
}

if [ $N02 -gt 0 ]; then
    submit_node "gpu02" "$TASKFILE_02" "$N02"
fi
if [ $N03 -gt 0 ]; then
    submit_node "gpu03" "$TASKFILE_03" "$N03"
fi

echo
echo "All batches submitted."
echo "Monitor: squeue -u \$USER -n mut52_${STUDY}_gpu02,mut52_${STUDY}_gpu03"
