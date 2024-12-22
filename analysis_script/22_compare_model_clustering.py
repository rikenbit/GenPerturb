#conda activate singlecell

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import PIL.Image as Image
import os





## studies
studies = [
"NormanWeissman2019_filtered_mixscape_exnp_train",
"ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
"ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
"ReplogleWeissman2022_rpe1_mixscape_exnp_train",
"JialongJiang2024_Myeloid_train",
"JialongJiang2024_CD4T_train",
"JialongJiang2024_CD8T_train",
"JialongJiang2024_B_cell_train",
"JuliaJoung2023_TFAtlas_train",
"Srivatsan2019_A549_train",
"Srivatsan2019_K562_train",
"Srivatsan2019_MCF7_train",
]

datanames = [
"Norman et al.",
"Replogle et al. essential",
"Replogle et al. gwps",
"Replogle et al. rpe1",
"Jialong et al. myeloid",
"Jialong et al. CD4T",
"Jialong et al. CD8T",
"Jialong et al. Bcell",
"Julia et al.",
"Srivatsan et al. A549",
"Srivatsan et al. K562",
"Srivatsan et al. MCF7",
]

name_replace = {
'enformer': 'Enformer',
'hyena_dna_tss': 'HyenaDNA TSS',
'hyena_dna_last': 'HyenaDNA last',
'nucleotide_tss': 'Nucleotide\nTransformer TSS',
'nucleotide_cls': 'Nucleotide\nTransformer CLS',
}


def heatmap(clust_stats, output=""):
    os.makedirs(f'across_study/compare_embedding', exist_ok=True)
    for comparison in set(clust_stats["Comparison"]):
        for metrics, cmap, vmax in zip(["NMI", "ARI", "FM"], ["viridis", "magma", "copper"], [0.6, 0.4, 0.4]):
            df = clust_stats.query(
                'metrics == @metrics & Comparison == @comparison'
            ).pivot_table(
                index="pretrained_model", columns="Study", values="value" ,sort=False
            ).round(2)
            plt.figure(figsize=(10/2.54, 6/2.54), dpi=300)
            plt.rcParams["font.size"] = 5.5
            plt.rcParams["axes.titlesize"] = 6
            plt.rcParams["axes.labelsize"] = 6
            g = sns.heatmap(df, annot=True, linewidths=.5,
                    vmax=vmax, vmin=0, cmap=cmap)
            for label in g.get_xticklabels():
                label.set_rotation(90)
            for label in g.get_yticklabels():
                label.set_rotation(0)
            plt.title(f"{comparison} ({metrics})")
            plt.tight_layout()
            plt.savefig(f'across_study/compare_embedding/clustering_stats_{comparison.replace(" ", "")}_{metrics}_{output}.svg')
            plt.clf()
            plt.close()


def plot_heatmap(suffixes, models, output):
    clust_stats = pd.DataFrame()
    for study_suffix, model in zip(suffixes, models):
        for study_name, dataname in zip(studies, datanames):
            study = f'{study_name}__{study_suffix}'
            clust_stat  = pd.read_csv(f"figures/{study}/embedding/clustering_metrics.txt", sep="\t")
            clust_stat["pretrained_model"] = model
            clust_stats = pd.concat([clust_stats, clust_stat])
    clust_stats['pretrained_model'] = clust_stats['pretrained_model'].replace(name_replace)
    heatmap(clust_stats, output)


suffixes = [
"enformer_transfer_epoch100_batch256_adamw5e3",
"hyena_dna_tss_transfer_epoch100_batch256_adamw5e3",
"nucleotide_transformer_tss_transfer_epoch100_batch256_adamw5e3",
]

models = [
"enformer",
"hyena_dna_tss",
"nucleotide_tss",
]

plot_heatmap(suffixes, models, "tss")



suffixes = [
"enformer_transfer_epoch100_batch256_adamw5e3",
"hyena_dna_last_transfer_epoch100_batch256_adamw5e3",
"nucleotide_transformer_cls_transfer_epoch100_batch256_adamw5e3",
]

models = [
"enformer",
"hyena_dna_last",
"nucleotide_cls",
]

plot_heatmap(suffixes, models, "cls")


