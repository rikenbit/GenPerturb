# conda activate enformer
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


enf_npy = np.load("data/enformer_heads_target4.npy")
enf_index = pd.read_csv("data/enformer_dataset_index.txt", sep="\t", names=["index", "cell_type"])
bed = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed",sep ="\t", usecols=[3], names=["Gene"])

def plot_scatter(res, study, outdir="adamw5e3"):
    color = sns.color_palette("deep")[0]
    corrs = res.iloc[:,2:].corr().iloc[0, 1:].round(3).to_list()
    cols = list(res.columns)[2:]
    fig, axes = plt.subplots(1, 2, figsize=(8/2.54, 5/2.54), dpi=300)
    col1 = cols[0]
    for i, (col2, corr) in enumerate(zip(cols[1:], corrs)):
        sns.scatterplot(x=col1, y=col2, data=res, ax=axes[i], s=3, color=color)
        axes[i].set_title(f"{col2}\nvs.\n{col1}")
        axes[i].annotate(str(f"r = {corr}"), xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=6, ha='left', va='top', color='black')
    plt.rcParams["font.size"] = 6
    plt.tight_layout()
    plt.savefig(f"across_study/compare_real_pred/{outdir}/{study}.svg")
    plt.clf()
    plt.close()

def compare_scatter(study, cell_type):
    label = enf_index.query('cell_type == @cell_type')["index"].values[0]
    enf_target_cell = enf_npy[:,:,label].sum(axis=1)
    enf_target_cell = enf_target_cell * 1e6 / enf_target_cell.sum()
    enf = np.log2(enf_target_cell + 1)
    enf = pd.concat([bed, pd.DataFrame({"Original enformer":enf})], axis=1).set_index("Gene")
    df = pd.read_csv(f"data/{study}.tsv", sep="\t", usecols=[0,1,2])
    df.rename(columns={df.columns[2]: "Real expression"}, inplace=True)
    pred = pd.DataFrame(np.load(f"prediction/{study}{condition}/prediction.npy"))
    pred = pred.rename(columns={0:"Transfer learning"}).loc[:,["Transfer learning"]]
    res = pd.concat([df, pred],axis=1)
    res = pd.merge(res, enf, left_on="Gene", right_index=True)
    plot_scatter(res.query('training == "test"'), study)


os.makedirs(f'across_study/compare_real_pred/{outdir}', exist_ok=True)
condition = "__enformer_transfer_epoch100_batch256_adamw5e3"

## K562 ##
studies = [
"NormanWeissman2019_filtered_mixscape_exnp_train",
"ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
"ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
"Srivatsan_2019_raw_K562_train",
]

## test
study = "NormanWeissman2019_filtered_mixscape_exnp_train"
cell_type = "CAGE:chronic myelogenous leukemia cell line:K562"


for study in studies:
    compare_scatter(study, "CAGE:chronic myelogenous leukemia cell line:K562")

## CD4T ##
study = "JialongJiang2024_CD4T_train"
compare_scatter(study, "CAGE:CD4+ T Cells,")

## CD8T ##
study = "JialongJiang2024_CD8T_train"
compare_scatter(study, "CAGE:CD8+ T Cells,")

## Bcell ##
study = "JialongJiang2024_B_cell_train"
compare_scatter(study, "CAGE:CD19+ B Cells,")

## Myeloid ##
study = "JialongJiang2024_Myeloid_train"
compare_scatter(study, "CAGE:Macrophage - monocyte derived,")


