#!/bin/bash
# ==============================================================================
# Fig 3d / S9 / 20: Peak evaluation with MartinRufino ATAC ground truth (light CPU)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

STUDY="${1:-MartinRufino2025_mixscape_exnp_train}"
MODEL="${2:-alphagenome}"
STUDY_SUFFIX="${3:-$(default_study_suffix "$MODEL")}"

pipeline_start_log "fig3_20_enhancer_auprc_atac_martin"
pipeline_activate_conda alphagenome

python scripts/attribution_evaluation/41_prepare_atac_peaks_martin.py

python scripts/attribution_evaluation/42_create_unified_peak_bed_martin_gtgenes.py \
    --study_name "${STUDY}" \
    --study_suffix "${STUDY_SUFFIX}" \
    --atac_root reference/martin_atac \
    --output_root cre_gtgenes \
    --attribution_suffix _gtgenes

python scripts/attribution_evaluation/43_compute_peak_scores.py \
    --study_name "${STUDY}" \
    --study_suffix "${STUDY_SUFFIX}" \
    --cre_root cre_gtgenes \
    --output_root cre_gtgenes \
    --attribution_filename_suffix _gtgenes \
    --attribution_score_source raw_h5

python scripts/attribution_evaluation/45_evaluate_auprc_deg_filtered_fc.py \
    --study martin \
    --study_name "${STUDY}" \
    --study_suffix "${STUDY_SUFFIX}" \
    --cre_root cre_gtgenes \
    --figures_root figures \
    --fc_thresholds 0,0.1,0.2,0.3,0.5 \
    --n_bootstrap 1000
