## conda activate singlecell
import pandas as pd
import numpy as np
import h5py


studies = [
    "NormanWeissman2019_filtered_mixscape_exnp",
    "ReplogleWeissman2022_K562_gwps_mixscape_exnp",
    "ReplogleWeissman2022_K562_essential_mixscape_exnp",
    "ReplogleWeissman2022_rpe1_mixscape_exnp",
    "Srivatsan2019_K562",
    "Srivatsan2019_MCF7",
    "Srivatsan2019_A549",
    "JialongJiang2024_B_cell",
    "JialongJiang2024_CD4T",
    "JialongJiang2024_CD8T",
    "JialongJiang2024_Myeloid",
    "MartinRufino2025_mixscape_exnp",
    "Wu2024_mixscape_exnp",
    "Shevade2025_K562_DMSO_mixscape_exnp",
    "Metzner2025_mixscape_exnp",
]


## Common ##
bed = pd.read_csv("fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed",
     sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "fold"])



## Enformer
npy = np.load("data/enformer_embedding.npy")
for study in studies:
    print(study)
    study_bed = pd.read_csv(f"fasta/{study}_train.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
    bed_index = pd.merge(bed, study_bed, on=["chr", "start", "end", "Gene"], how="left").dropna().index.to_list()
    assert len(bed_index) == len(study_bed), \
        f"  {study}: matched {len(bed_index)} genes but study BED has {len(study_bed)} entries"
    study_npy = npy[bed_index]
    print(f"  {study}: {len(study_bed)} genes, embedding shape {study_npy.shape}")
    with h5py.File(f'data/{study}_train_enformer.h5', 'w') as f:
        f.create_dataset('embedding', data=study_npy)


## Borzoi
npy = np.load("data/borzoi_embedding.npy")
for study in studies:
    print(study)
    study_bed = pd.read_csv(f"fasta/{study}_train.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
    bed_index = pd.merge(bed, study_bed, on=["chr", "start", "end", "Gene"], how="left").dropna().index.to_list()
    assert len(bed_index) == len(study_bed), \
        f"  {study}: matched {len(bed_index)} genes but study BED has {len(study_bed)} entries"
    study_npy = npy[bed_index]
    print(f"  {study}: {len(study_bed)} genes, embedding shape {study_npy.shape}")
    with h5py.File(f'data/{study}_train_borzoi.h5', 'w') as f:
        f.create_dataset('embedding', data=study_npy)

