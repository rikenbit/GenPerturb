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
MUTATION_PARTITION="${MUTATION_PARTITION:-${PARTITION_GPU}}"
MUTATION_GRES="${MUTATION_GRES:-${GPU_GRES}}"
MUTATION_NODES="${MUTATION_NODES:-}"
MAX_PER_NODE="${MAX_PER_NODE:-8}"
MEM_PER_JOB="${MEM_PER_JOB:-60G}"

mkdir -p "$LOGDIR"
mkdir -p "$RESULTDIR"

NTASKS=$(wc -l < "$TASKFILE")
echo "Study:       $STUDY"
echo "Total tasks: $NTASKS genes"
echo "Partition:   $MUTATION_PARTITION"
echo "GRES:        $MUTATION_GRES"
if [ -n "$MUTATION_NODES" ]; then
    echo "Nodes:       $MUTATION_NODES (max $MAX_PER_NODE jobs per node)"
else
    echo "Nodes:       scheduler selected (max $MAX_PER_NODE concurrent jobs)"
fi
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
        SBATCH_ARGS=(
            --parsable
            --job-name="mut52_${STUDY}_${NODE}" \
            --partition="$MUTATION_PARTITION" \
            --nodes=1 \
            --gres="$MUTATION_GRES" \
            --mem="$MEM_PER_JOB" \
            --cpus-per-task=2 \
            --array=1-${BATCH_SIZE}%${MAX_PER_NODE} \
            --output="${LOGDIR}/mutation_${NODE}_%A_%a.out" \
            --error="${LOGDIR}/mutation_${NODE}_%A_%a.err" \
            --export=ALL,TASK_OFFSET=${OFFSET},STUDY=${STUDY},NODE_TASKFILE=${NODE_TASKFILE},PROJECT_ROOT=${CWD},CONDA_SH=${CONDA_SH}
        )
        [ "$NODE" != "scheduler" ] && SBATCH_ARGS+=(--nodelist="$NODE")

        JOBID=$(sbatch "${SBATCH_ARGS[@]}" \
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

if [ -n "$MUTATION_NODES" ]; then
    IFS=',' read -r -a NODE_ARRAY <<< "$MUTATION_NODES"
else
    NODE_ARRAY=("scheduler")
fi

for i in "${!NODE_ARRAY[@]}"; do
    NODE="${NODE_ARRAY[$i]}"
    NODE_TASKFILE="${STUDY_DIR}/tasks_${NODE}.txt"
    awk -v n="${#NODE_ARRAY[@]}" -v r="$i" '((NR - 1) % n) == r' "$TASKFILE" > "$NODE_TASKFILE"
    NODE_NTASKS=$(wc -l < "$NODE_TASKFILE")
    echo "  -> ${NODE}: ${NODE_NTASKS} tasks (${NODE_TASKFILE})"
    if [ "$NODE_NTASKS" -gt 0 ]; then
        submit_node "$NODE" "$NODE_TASKFILE" "$NODE_NTASKS"
    fi
done

echo
echo "All batches submitted."
echo "Monitor: squeue -u \$USER -n mut52_${STUDY}_*"
