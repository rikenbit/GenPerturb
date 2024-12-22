## conda activate singlecell
import pandas as pd
import numpy as np
import random
import h5py


def merge_and_save(bed, df, study, suffix=""):
    merge = pd.merge(bed, df, left_on="Gene", right_index=True)
    merge.iloc[:,:7].to_csv(f'fasta/{study}_train{suffix}.bed', index=False, header=False, sep="\t")
    merge.set_index("Gene").iloc[:,5:].to_csv(f'data/{study}_train{suffix}.tsv', sep="\t")
    with h5py.File(f'data/{study}_train{suffix}.h5', 'w') as f:
        f.create_dataset('embedding', data=npy[merge.index.to_list()])


def process_data(bed, npy, study, cpm_threshold=2, suffix="", gene_filter=True):
    if gene_filter:
        df_cpm = pd.read_csv(f"data/{study}_cpm.tsv", sep="\t", index_col=[0]).astype("float32")
        values_per_gene= df_cpm.apply(lambda row: np.percentile(row, 100), axis=1)
        df_cpm = df_cpm[values_per_gene > cpm_threshold]
    else:
        df_cpm  = pd.read_csv(f"data/{study}_cpm.tsv", sep="\t", index_col=[0]).astype("float32")
    df_cpm = np.log2(df_cpm + 1)
    merge_and_save(bed, df_cpm, study, suffix=suffix)


## Common ##
bed = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "training"])
npy = np.load("data/enformer_embedding.npy")

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


for study in studies:
    process_data(bed, npy, study)


