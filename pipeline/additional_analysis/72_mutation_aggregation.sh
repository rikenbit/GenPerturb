#!/bin/bash
# ==============================================================================
# 72: Matched in-silico mutation aggregation
#   Source data for Fig. 4f and Fig. S9b
# Tertile boundaries are the fixed study-specific values; do not reselect them.
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY_KEY="${1:-Martin_matched}"
OUTROOT="${OUTROOT:-results/mutation_effects}"

case "$STUDY_KEY" in
    Martin_matched)
        TSS_BED="fasta/MartinRufino2025_mixscape_exnp_train.bed"
        LOW=0.0332274039586385
        HIGH=0.05953089396158846
        ;;
    Norman_matched)
        TSS_BED="fasta/NormanWeissman2019_filtered_mixscape_exnp_train.bed"
        LOW=0.05784606933593749
        HIGH=0.12850379943847642
        ;;
    *)
        echo "Unknown study key: $STUDY_KEY" >&2
        exit 1
        ;;
esac

pipeline_start_log "step72_mutation_${STUDY_KEY}"
pipeline_activate_conda singlecell
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/additional_analysis/fig4f_figS9b_mutation_aggregation.py \
    --data-dir "attribution_analysis/insilico_mutation/${STUDY_KEY}" \
    --tss-bed "${TSS_BED}" \
    --context-length 1048576 \
    --low-cutoff "${LOW}" --high-cutoff "${HIGH}" \
    --cutoff-source 'fixed study-specific attribution tertile boundaries' \
    --out "${OUTROOT}/${STUDY_KEY}"
