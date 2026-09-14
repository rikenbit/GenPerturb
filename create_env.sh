#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
create_env.sh is retained only for backward compatibility.

Publication environment specifications now live in:

  paper_repository/environments/

Create an environment from paper_repository/ with, for example:

  conda env create -f environments/alphagenome.yml

See environments/README.md for the production evidence, CUDA/PyTorch notes,
VCS commit pins, runtime token setup, GimmeMotifs reference-genome registration,
and the TF-MoDISco post-install patch.
MSG
