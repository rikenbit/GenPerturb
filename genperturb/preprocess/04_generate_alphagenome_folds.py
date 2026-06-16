## conda activate alphagenome
## Generate AlphaGenome all_regions BED file with fold labels (fold0-fold7)
## Output: fasta/alphagenome/all_regions.bed
## 20_alphagenome_embedding.py extracts ALL_FOLDS and FOLD_0..3.


import os
import pandas as pd
from alphagenome.data import fold_intervals
from alphagenome.models import dna_client

OUTPUT_DIR = "fasta/alphagenome"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_regions = fold_intervals.get_fold_intervals(
    model_version=dna_client.ModelVersion.ALL_FOLDS,
    organism=dna_client.Organism.HOMO_SAPIENS,
    subset=fold_intervals.Subset.TRAIN,
)

all_regions = all_regions.sort_values(["chromosome", "start"]).reset_index(drop=True)
all_regions.to_csv(f"{OUTPUT_DIR}/all_regions.bed", sep="\t", index=False, header=False)
print(f"Saved {OUTPUT_DIR}/all_regions.bed: {len(all_regions)} intervals")
