#!/bin/bash
# ==============================================================================
# Fig 4 / 40: pySCENIC GRN + master regulator upset (S12)
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

J1=$(sbatch --parsable \
    -J fig4_40_scenic \
    -o log/fig4_40_scenic.out \
    -e log/fig4_40_scenic.err \
    --partition="${PARTITION_CPU}" --cpus-per-task=32 --mem="${MEM_GPU}" --time=24:00:00 \
    <<SCRIPT
#!/bin/bash
source ${CONDA_SH}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="\${TMPDIR:-/tmp}/numba_cache_\${SLURM_JOB_ID:-manual}"
mkdir -p "\${NUMBA_CACHE_DIR}"
conda activate singlecell
python scripts/immune_differentiation/31a_run_scenic.py
SCRIPT
)

J2=$(sbatch --parsable \
    --dependency=afterok:${J1} \
    -J fig4_40_upset \
    -o log/fig4_40_master_regulator.out \
    -e log/fig4_40_master_regulator.err \
    --partition="${PARTITION_CPU}" --mem="${MEM_LIGHT}" \
    <<SCRIPT
#!/bin/bash
source ${CONDA_SH}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="\${TMPDIR:-/tmp}/numba_cache_\${SLURM_JOB_ID:-manual}"
mkdir -p "\${NUMBA_CACHE_DIR}"
conda activate singlecell
python scripts/immune_differentiation/32_master_regulator_upset.py
SCRIPT
)
echo "Submitted: ${J1} -> ${J2}"
