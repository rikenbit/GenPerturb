#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Create the required Conda environments from the repository root. For example:

  conda env create -f environments/alphagenome.yml

See environments/README.md for the analysis-to-environment mapping,
TF-MoDISco post-install step and GimmeMotifs reference-genome registration.
MSG
