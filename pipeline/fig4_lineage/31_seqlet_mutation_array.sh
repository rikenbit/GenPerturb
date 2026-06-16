#!/bin/bash
# ==============================================================================
# Fig 4 / 31: In-silico seqlet mutation per marker gene (S11)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY_FULL="${STUDY_FULL:-NormanWeissman2019_filtered_mixscape_exnp_train__$(default_study_suffix alphagenome)}"
SCRIPT_PY="scripts/immune_differentiation/21a_insilico_seqlet_mutation.py"
OUTDIR="figures/${STUDY_FULL}/seqlet_mutation/mutation_predictions"
GENES=(HBG1 HBG2 HBZ HBA1 HBA2 GYPA ITGAM CSF3R LST1)
MAX_CONCURRENT="${MAX_CONCURRENT:-9}"

mkdir -p "$OUTDIR"

JIDS=()
for i in "${!GENES[@]}"; do
    GENE="${GENES[$i]}"
    DEP_ARG=""
    if [ "$i" -ge "$MAX_CONCURRENT" ]; then
        DEP_IDX=$((i - MAX_CONCURRENT))
        DEP_ARG="--dependency=afterany:${JIDS[$DEP_IDX]}"
    fi

    JID=$(sbatch --parsable ${DEP_ARG} \
        -J "fig4_31_${GENE}" \
        -o "log/fig4_31_seqlet_mut_${GENE}.out" \
        -e "log/fig4_31_seqlet_mut_${GENE}.err" \
        --partition="${PARTITION_GPU}" --gres="${GPU_GRES}" --mem="${MEM_GPU}" \
        --time=24:00:00 \
        <<SCRIPT
#!/bin/bash
source ${CONDA_SH}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
conda activate alphagenome
python ${SCRIPT_PY} --gene ${GENE} --n-seeds 5 --max-seqlets 50 --outdir ${OUTDIR}
SCRIPT
)
    JIDS+=("$JID")
    echo "  ${GENE}: ${JID}${DEP_ARG:+ dep=${DEP_ARG}}"
done
