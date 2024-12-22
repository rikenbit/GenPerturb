#!/bin/bash
# A script for submitting jobs using either qsub or sbatch.
# The commands depend on the individual server.


source ~/.bashrc

CWD="/path_to/GenPerturb"
cd $CWD

conda activate singlecell
python genperturb/preprocess/11_pertpy.py

conda activate enformer
srun python genperturb/preprocess/20_enformer_embedding.py
srun python genperturb/preprocess/24_enformer_embedding_masked.py

conda activate hyena_dna
srun python genperturb/preprocess/21_hyenadnaget_embedding.py

conda activate nucleotide
srun python genperturb/preprocess/22_nucleotide_transformer_embedding.py
