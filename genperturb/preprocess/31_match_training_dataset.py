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
    "JuliaJoung2023_TFAtlas",
    "JialongJiang2024_B_cell",
    "JialongJiang2024_CD4T",
    "JialongJiang2024_CD8T",
    "JialongJiang2024_Myeloid"
]


## Common ##
bed = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed",
     sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])


## HyenaDNA ##
for emb in ["tss", "last", "mean"]:
    npy = np.load(f"data/hyena_embedding_160k_{emb}.npy")
    for study in studies:
        print(study)
        model = "hyena"
        study_bed = pd.read_csv(f"fasta/{study}_train.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
        bed_index = pd.merge(bed, study_bed, on=["chr", "start", "end", "Gene"], how="left").dropna().index.to_list()
        study_npy = npy[bed_index]
        len(study_bed)
        study_npy.shape
        with h5py.File(f'data/{study}_train_{model}_{emb}.h5', 'w') as f:
            f.create_dataset('embedding', data=study_npy)


## Nucleotide transformer ##
for emb in ["tss", "cls", "mean"]:
    npy = np.load(f"data/nt_embedding_v2_500m_{emb}.npy")
    for study in studies:
        print(study)
        value = "tpm"
        model = "nt"
        study_bed = pd.read_csv(f"fasta/{study}_train.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
        bed_index = pd.merge(bed, study_bed, on=["chr", "start", "end", "Gene"], how="left").dropna().index.to_list()
        study_npy = npy[bed_index]
        len(study_bed)
        study_npy.shape
        with h5py.File(f'data/{study}_train_{model}_{emb}.h5', 'w') as f:
            f.create_dataset('embedding', data=study_npy)



## enformer with masked condition ##
npy = np.load(f"data/enformer_embedding_masked.npy")
for study in studies:
    print(study)
    model = "enformer"
    study_bed = pd.read_csv(f"fasta/{study}_train.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
    bed_index = pd.merge(bed, study_bed, on=["chr", "start", "end", "Gene"], how="left").dropna().index.to_list()
    study_npy = npy[bed_index]
    len(study_bed)
    study_npy.shape
    with h5py.File(f'data/{study}_train_{model}_masked.h5', 'w') as f:
        f.create_dataset('embedding', data=study_npy)




