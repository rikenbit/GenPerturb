#!/bin/bash
# ==============================================================================
# Fig 5: Drug mechanism analysis
# ==============================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
cd "${PROJECT_ROOT}"

pipeline_start_log fig5_51_drug_mechanism
pipeline_activate_conda singlecell

echo "=== NR3C1 ==="
python scripts/drug_mechanism/00_pubchem_glucocorticoid_screen.py

python scripts/drug_mechanism/01b_glucocorticoid_rank_NR3C1.py
python scripts/drug_mechanism/02_motif_vs_tf_NR3C1_rank_pct.py

echo "=== compound-motif scatter ==="
python scripts/drug_mechanism/03_compound_motif_scatter.py
