import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



ag_npy = np.load("data/alphagenome_prediction.npy", mmap_mode='r')

targets = pd.read_csv("fasta/borzoi/targets_human.txt", sep="\t", index_col=0)

bed = pd.read_csv(
    "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed",
    sep="\t", usecols=[3], names=["Gene"]
)



def find_cage_tracks(targets, cell_type_keyword):
    cage_mask = targets['description'].str.startswith('CAGE:')
    keyword_mask = targets['description'].str.contains(cell_type_keyword, regex=False)
    matched = targets[cage_mask & keyword_mask]

    results = []
    seen = set()

    for idx, row in matched.iterrows():
        desc = row['description']
        if desc in seen:
            continue
        seen.add(desc)

        same_desc = matched[matched['description'] == desc]
        plus_tracks = same_desc[same_desc['identifier'].str.endswith('+')]
        minus_tracks = same_desc[same_desc['identifier'].str.endswith('-')]

        if len(plus_tracks) > 0 and len(minus_tracks) > 0:
            results.append((desc, plus_tracks.index[0], minus_tracks.index[0]))

    return results



EDGE_COLOR = "#B0B0B0"
LINE_COLOR = "#666666"


def plot_scatter(res, study, label, outdir="alphagenome"):
    color = sns.color_palette("deep")[0]
    corrs = res.iloc[:, 2:].corr().iloc[0, 1:].round(3).to_list()
    cols = list(res.columns)[2:]
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "xtick.color": EDGE_COLOR,
        "ytick.color": EDGE_COLOR,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(18 / 2.54, 10 / 2.54), dpi=300)
    col1 = cols[0]
    for i, (col2, corr) in enumerate(zip(cols[1:], corrs)):
        sns.scatterplot(x=col1, y=col2, data=res, ax=axes[i], s=4, color=color)
        axes[i].set_title(f"{col2}\nvs.\n{col1}", fontsize=12)
        axes[i].set_ylabel(col2.replace(" (", "\n("))
        axes[i].set_box_aspect(1)
        axes[i].annotate(
            f"r = {corr}", xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top', color='black'
        )
    plt.tight_layout()
    plt.savefig(f"across_study/compare_real_pred/{outdir}/{study}_{label}.svg")
    plt.clf()
    plt.close()



def compare_scatter(study, cell_type_keyword, condition, outdir="alphagenome"):
    tracks = find_cage_tracks(targets, cell_type_keyword)

    if not tracks:
        print(f"  WARNING: No CAGE tracks found for '{cell_type_keyword}'")
        return

    for desc, plus_idx, minus_idx in tracks:
        ag_target = ag_npy[:, :, plus_idx].sum(axis=1) + ag_npy[:, :, minus_idx].sum(axis=1)

        total = ag_target.sum()
        if total > 0:
            ag_target = ag_target * 1e6 / total
        ag = np.log2(ag_target + 1)

        ag_df = pd.concat(
            [bed, pd.DataFrame({"AlphaGenome (pretrained)": ag})], axis=1
        ).set_index("Gene")

        df = pd.read_csv(f"data/{study}.tsv", sep="\t", usecols=[0, 1, 2])
        df.rename(columns={df.columns[2]: "Real expression"}, inplace=True)

        pred = pd.DataFrame(np.load(f"prediction/{study}{condition}/prediction.npy"))
        pred = pred.rename(columns={0: "GenPerturb (AlphaGenome transfer)"}).loc[:, ["GenPerturb (AlphaGenome transfer)"]]

        res = pd.concat([df, pred], axis=1)
        res = pd.merge(res, ag_df, left_on="Gene", right_index=True)

        label = (desc.replace("CAGE:", "")
                 .replace(" ", "_").replace(",", "").replace(":", "_"))

        plot_scatter(res.query('training == "test"'), study, label, outdir=outdir)
        print(f"  Plotted: {study} - {desc} (tracks {plus_idx}+/{minus_idx}-)")



outdir = "alphagenome"
condition = "__alphagenome_transfer_epoch100_batch256_adamw5e3"

os.makedirs(f'across_study/compare_real_pred/{outdir}', exist_ok=True)

## K562 ##
print("=== K562 ===")
k562_studies = [
    "NormanWeissman2019_filtered_mixscape_exnp_train",
    "ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
    "ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
    "Srivatsan2019_K562_train",
]
for study in k562_studies:
    compare_scatter(study, "chronic myelogenous leukemia cell line:K562", condition, outdir)

## RPE1 ##
print("=== RPE1 ===")
compare_scatter(
    "ReplogleWeissman2022_rpe1_mixscape_exnp_train",
    "Retinal Pigment Epithelial Cells,",
    condition, outdir
)

## A549 ##
print("=== A549 ===")
compare_scatter(
    "Srivatsan2019_A549_train",
    "lung adenocarcinoma cell line:A549",
    condition, outdir
)

## MCF7 ##
print("=== MCF7 ===")
compare_scatter(
    "Srivatsan2019_MCF7_train",
    "breast carcinoma cell line:MCF7",
    condition, outdir
)

## CD4T ##
print("=== CD4T ===")
compare_scatter(
    "JialongJiang2024_CD4T_train",
    "CD4+ T Cells,",
    condition, outdir
)

## CD8T ##
print("=== CD8T ===")
compare_scatter(
    "JialongJiang2024_CD8T_train",
    "CD8+ T Cells",
    condition, outdir
)

## B cell ##
print("=== B cell ===")
compare_scatter(
    "JialongJiang2024_B_cell_train",
    "CD19+ B Cells",
    condition, outdir
)

## Myeloid ##
print("=== Myeloid ===")
compare_scatter(
    "JialongJiang2024_Myeloid_train",
    "Macrophage - monocyte derived,",
    condition, outdir
)

## Erythroid (Martin-Rufino) ##
print("=== Erythroid (Martin-Rufino) ===")
compare_scatter(
    "MartinRufino2025_mixscape_exnp_train",
    "CD34 cells differentiated to erythrocyte lineage",
    condition, outdir
)
