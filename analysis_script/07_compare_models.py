# conda activate enformer
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import PIL.Image as Image
import sys
import os


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

xlabels = [
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

pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"hyena_dna_mean",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
"nucleotide_transformer_mean",
]


name_replace = {
'enformer': 'Enformer',
'hyena_dna_tss': 'HyenaDNA TSS',
'hyena_dna_last': 'HyenaDNA last',
'nucleotide_transformer_tss': 'Nucleotide Transformer TSS',
'nucleotide_transformer_cls': 'Nucleotide Transformer CLS',
}


epoch = 100
batch = 256

cor_acpertss_dict = {}
cor_acgeness_dict = {}
cor_acpertss_fc_dict = {}
cor_acgeness_fc_dict = {}

for pretrained_model in pretrained_models:
    cor_acpertss_dict[pretrained_model] = pd.DataFrame()
    cor_acgeness_dict[pretrained_model] = pd.DataFrame()
    cor_acpertss_fc_dict[pretrained_model] = pd.DataFrame()
    cor_acgeness_fc_dict[pretrained_model] = pd.DataFrame()
    for study_name, xlabel in zip(studies, xlabels):
        study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
        cor_acperts = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
        cor_acgenes = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_genes.txt", sep="\t")
        cor_acperts_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation_across_perts.txt", sep="\t")
        cor_acgenes_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation_across_genes.txt", sep="\t")
        cor_acperts["study"] = xlabel
        cor_acgenes["study"] = xlabel
        cor_acperts_fc["study"] = xlabel
        cor_acgenes_fc["study"] = xlabel
        cor_acperts["model"] = pretrained_model
        cor_acgenes["model"] = pretrained_model
        cor_acperts_fc["model"] = pretrained_model
        cor_acgenes_fc["model"] = pretrained_model
        cor_acpertss_dict[pretrained_model] = pd.concat([cor_acpertss_dict[pretrained_model], cor_acperts], axis=0)
        cor_acgeness_dict[pretrained_model] = pd.concat([cor_acgeness_dict[pretrained_model], cor_acgenes], axis=0)
        cor_acpertss_fc_dict[pretrained_model] = pd.concat([cor_acpertss_fc_dict[pretrained_model], cor_acperts_fc], axis=0)
        cor_acgeness_fc_dict[pretrained_model] = pd.concat([cor_acgeness_fc_dict[pretrained_model], cor_acgenes_fc], axis=0)



## merge stats across genes ##
os.makedirs(f'across_study/compare_models/merge_models', exist_ok=True)

def plot_barplot_by_exp(df, pretrained_models, stats="Correlation", value="", output="_across_perturbations", yliml=0, ylimh=1):
    plt.figure(figsize=(12/2.54, 6/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    ax = sns.barplot(data=df, x="study", y=stats, hue='model', hue_order=pretrained_models, palette="deep")
    ax.set_ylim(yliml, ylimh)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Pretrained model")
    plt.xticks(rotation=60, ha='right', rotation_mode='anchor')
    plt.title(f'{stats} across genes')
    plt.tight_layout()
    plt.savefig(f'across_study/compare_models/merge_models/{stats}_{output}_{value}.svg')
    plt.clf()
    plt.close()

group = [
["enformer", "hyena_dna_tss", "nucleotide_transformer_tss"],
["enformer", "hyena_dna_last", "nucleotide_transformer_cls"],
]

for pretrained_models in group:
    value = pretrained_models[2].split("_")[-1]
    cor_acpertss_fc_merge = pd.concat([cor_acpertss_fc_dict[i] for i in pretrained_models])
    cor_acpertss_merge = pd.concat([cor_acpertss_dict[i] for i in pretrained_models])
    cor_acpertss_merge["model"] = cor_acpertss_merge["model"].replace(name_replace)
    cor_acgeness_merge = pd.concat([cor_acgeness_dict[i] for i in pretrained_models])
    cor_acgeness_merge["model"] = cor_acgeness_merge["model"].replace(name_replace)
    pretrained_models = [name_replace.get(i, i) for i in pretrained_models]
    plot_barplot_by_exp(cor_acgeness_merge.query('training == "test"'), pretrained_models, stats="Correlation", output="across_genes", value=value)


#### Representative sampler scatter plot ####
## select genes ##
cor_summary = pd.pivot_table(cor_acpertss_fc_merge.query('study == "Norman_K562_CRISPRa" & training == "test"'), index="Gene", columns="model", values="Correlation")
cor_summary["median"] = cor_summary.median(axis=1)
cor_summary["sub"] = cor_summary["enformer"] - cor_summary["hyena_dna"]
high_median_genes = cor_summary.sort_values("median").tail(3).index.to_list()
different_genes = cor_summary.query('hyena_dna > 0').sort_values("sub").tail(3).index.to_list()


ctrls = [
'Norman.NT',
'Replogle_essential.NT',
'Replogle_gwps.NT',
'Replogle_rpe1.NT',
'Jialong.CONTROL_CD3.Myeloid',
'Jialong.CONTROL_CD3.CD4T',
'Jialong.CONTROL_CD3.CD8T',
'Jialong.CONTROL_CD3.B_cell',
'Julia.NT',
'Srivatsan_A549.Vehicle_Vehicle_0nM',
'Srivatsan_K562.Vehicle_Vehicle_0nM', 
'Srivatsan_MCF7.Vehicle_Vehicle_0nM',
]

directory_names = f"across_study/compare_models/correlation_across_genes"
for ctrl, study, xlabel in zip(ctrls, studies, xlabels):
    image_paths = []
    for pretrained_model in ["enformer", "hyena_dna", "nucleotide_transformer"]:
        file_dir = f'figures/{study}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3/scatterplot/{ctrl}.png'
        image_paths.append(file_dir)
    out_dir  = f"{directory_names}/{xlabel}"

directory_names = f"across_study/compare_models/correlation_across_perturbations"
for study, xlabel in zip(studies, xlabels):
    for gene in high_median_genes + different_genes:
        for fc in ["", "_fc"]:
            image_paths = []
            for pretrained_model in ["enformer", "hyena_dna", "nucleotide_transformer"]:
                file_dir = f'figures/{study}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3/scatterplot{fc}/{gene}.png'
                image_paths.append(file_dir)
            out_dir  = f"{directory_names}/{xlabel}"


### scatter plot comparing models ###
def plot_scatter(data, i, j, l, train, study):
    data = data.query('(training == @train) & (study == @study)')
    median_rx = data[f"Correlation_x"].median().round(3)
    median_ry = data[f"Correlation_y"].median().round(3)
    os.makedirs(f'across_study/compare_models/correlation_{l}/{study.replace(" ", "")}/', exist_ok=True)
    plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
    plt.rcParams["font.size"] = 6
    plt.rcParams['axes.linewidth'] = 0.5
    ax = sns.scatterplot(x="Correlation_x", y="Correlation_y", data=data, c="slateblue", s=1.5)
    ax.plot([-1, 1], [-1, 1], color="black", linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.annotate(str(median_ry), xy=(0.05, 0.95), xycoords='axes fraction', 
            fontsize=5, ha='left', va='top', color='black')
    ax.annotate(str(median_rx), xy=(0.95, 0.05), xycoords='axes fraction', 
            fontsize=5, ha='right', va='bottom', color='black')
    plt.title(f'Correlation {l.replace("_", " ")}\n{study}\n({train})', fontsize=5)
    plt.xlabel(i)
    plt.ylabel(j)
    plt.tight_layout()
    plt.savefig(f'across_study/compare_models/correlation_{l}/{study.replace(" ", "")}/scatterplot.{i}.{j}.{study.replace(" ", "")}.{train}.svg')
    plt.clf()
    plt.close()


conditions = [
["enformer", "hyena_dna_tss"],
["enformer", "hyena_dna_last"],
["enformer", "nucleotide_transformer_tss"],
["enformer", "nucleotide_transformer_cls"],
["hyena_dna_tss", "nucleotide_transformer_tss"],
["hyena_dna_last", "nucleotide_transformer_cls"],
]

for i,j in conditions:
    for k,l in zip([cor_acpertss_dict, cor_acgeness_dict], ["across_perturbations", "across_genes"]):
        data = pd.merge(k[i].loc[:, ["Gene", "training", "study", "Correlation"]], k[j].loc[:, ["Gene", "training", "study", "Correlation"]], on=["Gene", "training", "study"])
        for train in ['train', 'val', 'test']:
            for study in set(data["study"]):
                plot_scatter(data, i, j, l, train, study)

