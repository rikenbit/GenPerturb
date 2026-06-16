## conda activate singlecell
import argparse
import pandas as pd
import numpy as np
import h5py


# AlphaGenome fold -> train/val/test mapping per model version
FOLD_MAPPING = {
    "all_folds": {"valid": {"fold0"}, "test": {"fold1"}},
    "fold_0":    {"valid": {"fold0"}, "test": {"fold1"}},
    "fold_1":    {"valid": {"fold3"}, "test": {"fold4"}},
    "fold_2":    {"valid": {"fold2"}, "test": {"fold5"}},
    "fold_3":    {"valid": {"fold6"}, "test": {"fold7"}},
}


def fold_to_training(fold_label, model_version):
    mapping = FOLD_MAPPING[model_version]
    if fold_label in mapping["valid"]:
        return "val"
    elif fold_label in mapping["test"]:
        return "test"
    else:
        return "train"


def merge_and_save(bed, df, study, npy, suffix="", h5_suffix=None):
    if h5_suffix is None:
        h5_suffix = suffix
    merge = pd.merge(bed, df, left_on="Gene", right_index=True)
    merge.iloc[:,:7].to_csv(f'fasta/{study}_train{suffix}.bed', index=False, header=False, sep="\t")
    merge.set_index("Gene").iloc[:,5:].to_csv(f'data/{study}_train{suffix}.tsv', sep="\t")
    with h5py.File(f'data/{study}_train{h5_suffix}.h5', 'w') as f:
        f.create_dataset('embedding', data=npy[merge.index.to_list()])


def process_data(bed, npy, study, cpm_threshold=2, suffix="", h5_suffix=None, gene_filter=True):
    if gene_filter:
        df_cpm = pd.read_csv(f"data/{study}_cpm.tsv", sep="\t", index_col=[0]).astype("float32")
        values_per_gene= df_cpm.apply(lambda row: np.percentile(row, 100), axis=1)
        df_cpm = df_cpm[values_per_gene > cpm_threshold]
    else:
        df_cpm  = pd.read_csv(f"data/{study}_cpm.tsv", sep="\t", index_col=[0]).astype("float32")
    df_cpm = np.log2(df_cpm + 1)
    merge_and_save(bed, df_cpm, study, npy, suffix=suffix, h5_suffix=h5_suffix)


parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="all_folds",
                    choices=["all_folds", "fold_0", "fold_1", "fold_2", "fold_3"])
args = parser.parse_args()

model_version = args.model_version

bed = pd.read_csv("fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand", "fold"])
bed["training"] = bed["fold"].apply(lambda x: fold_to_training(x, model_version))
bed = bed[["chr", "start", "end", "Gene", "score", "strand", "training"]]

model_name = "alphagenome" if model_version == "all_folds" else f"alphagenome_{model_version}"
h5_suffix = f"_{model_name}"

npy = np.load(f"data/{model_name}_embedding.npy")

if model_version == "all_folds":
    output_suffix = ""
else:
    output_suffix = h5_suffix

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


import os
for study in studies:
    cpm_path = f"data/{study}_cpm.tsv"
    if not os.path.exists(cpm_path):
        print(f"  Skipping {study}: {cpm_path} not found")
        continue
    process_data(bed, npy, study, suffix=output_suffix, h5_suffix=h5_suffix)
