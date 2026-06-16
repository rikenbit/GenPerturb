#!/bin/bash
# ==============================================================================
# Shared environment for pipeline/* scripts
# ==============================================================================
# Source this from every pipeline script:
#   source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
#
# Provides:
#   PROJECT_ROOT          repo root
#   CONDA_SH              conda profile script
#   PARTITION_GPU         default GPU partition
#   PARTITION_CPU         default CPU partition
#   GPU_GRES              default GPU gres spec
#   MEM_GPU / MEM_HEAVY / MEM_LIGHT
#   STUDIES[] / SHORT[]   default study list and short codes
#   DEFAULT_MODELS[]      pretrained models
#   default_study_suffix  function: echoes "<model>_transfer_epoch100_batch256_adamw5e3"
# ==============================================================================

set -euo pipefail

# Resolve PROJECT_ROOT one level up from this file (pipeline/_common.sh -> repo)
if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONDA_SH="${CONDA_SH:-/lustre/home/tshii/miniconda3/etc/profile.d/conda.sh}"
export CONDA_SH

PARTITION_GPU="${PARTITION_GPU:-h200-long}"
PARTITION_CPU="${PARTITION_CPU:-mi300x-long}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
MEM_GPU="${MEM_GPU:-240G}"
MEM_HEAVY="${MEM_HEAVY:-128G}"
MEM_LIGHT="${MEM_LIGHT:-64G}"

# Canonical study list (order matches SHORT[])
STUDIES=(
    "NormanWeissman2019_filtered_mixscape_exnp_train"
    "ReplogleWeissman2022_K562_essential_mixscape_exnp_train"
    "ReplogleWeissman2022_K562_gwps_mixscape_exnp_train"
    "ReplogleWeissman2022_rpe1_mixscape_exnp_train"
    "JialongJiang2024_Myeloid_train"
    "JialongJiang2024_CD4T_train"
    "JialongJiang2024_CD8T_train"
    "JialongJiang2024_B_cell_train"
    "Srivatsan2019_A549_train"
    "Srivatsan2019_K562_train"
    "Srivatsan2019_MCF7_train"
    "MartinRufino2025_mixscape_exnp_train"
)
SHORT=(
    "norman" "repl_ess" "repl_gwps" "repl_rpe1"
    "jj_mye" "jj_cd4t" "jj_cd8t" "jj_bcell"
    "sri_a549" "sri_k562" "sri_mcf7"
    "mr_all"
)

DEFAULT_MODELS=("alphagenome" "borzoi" "enformer")

default_study_suffix() {
    local model="$1"
    echo "${model}_transfer_epoch100_batch256_adamw5e3"
}

mkdir -p "${PROJECT_ROOT}/log"

pipeline_start_log() {
    local logtag="$1"
    mkdir -p "${PROJECT_ROOT}/log"
    exec > >(tee "${PROJECT_ROOT}/log/${logtag}.log") 2>&1
    echo "=== ${logtag} ==="
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "CWD: ${PROJECT_ROOT}"
}

pipeline_activate_conda() {
    local conda_env="$1"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    set +u
    source "${CONDA_SH:?CONDA_SH is required}"
    conda activate "${conda_env}"
    set -u
}
