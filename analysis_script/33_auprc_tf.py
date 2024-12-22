# conda activate enformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import PIL.Image as Image
import pybedtools
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os


study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman_K562_CRISPRa"

pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
]


tss_path = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed"
tss = pd.read_csv(tss_path, sep="\t", usecols=[0,1,2,3], names=["chr", "start", "end", "gene"])
tss["start"] = tss["start"] - 1000
tss["end"] = tss["end"] + 1000
tss_out = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.1000.bed"
tss.to_csv(tss_out, index=False, header=False, sep="\t")
tss = pybedtools.BedTool(tss_out)

tfs = ['HNF4A', 'TP73', 'IRF1', 'AHR', 'CEBPA', 'SPI1', 'KMT2A', 'PRDM1', 'CEBPB', 'SNAI1', 'JUN', 'ETS2', 'FOXA1', 'EGR1']
peak_list = pd.read_csv("reference/chipatlas/download_data_list.tsv", sep="\t")


def load_exp(study):
    df = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
    pred = np.load(f"prediction/{study}/prediction.npy")
    df2 = pd.DataFrame(pred)
    df2.columns = df.columns[1:]
    df2.index = df.index
    ctrl = df2.columns[0]
    df3 = (df2.T - df2[ctrl]).T.drop(ctrl, axis=1).copy()
    df_corr = pd.read_csv(f"figures/{study}/cor_matrix/correlation_across_perts.txt", sep="\t")
    return df3, df_corr


def plot_barplot(summary_stats, study, pretrained_model, y):
    pallete = sns.color_palette("Dark2") + sns.color_palette("Set1_r")
    plt.figure(figsize=(15/2.54, 9/2.54), dpi=300)
    plt.rcParams["font.size"] = 5
    sns.set_theme(style="whitegrid")
    g = sns.barplot(data=summary_stats, hue="pert", y=y, palette=pallete)
    g.set_title(f"{pretrained_model}\nAUPRC\nAttribution - ChIP-seq")
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=7)
    plt.tight_layout()
    plt.savefig(f"figures/{study}/attribution/chip_auprc/{pretrained_model}_{y}.svg")
    plt.close()


for pretrained_model in pretrained_models:
    study = f'{study_name}__{pretrained_model}_transfer_epoch100_batch256_adamw5e3'
    df3, df_corr = load_exp(study)
    peaks, attrs, ovlap, stats = {}, {}, {}, {}
    peak_pivot = pd.DataFrame()
    for tf in tfs:
    #for tf in ["HNF4A"]:
        for srx in peak_list.query('Target == @tf')["SRX"].to_list():
            print(f"{tf} {srx}")
            peak = pybedtools.BedTool(f"reference/chipatlas/{srx}.05.bed")
            peak_ex = peak.to_dataframe()
            peak_ex["start"] = peak_ex["start"].where(peak_ex["start"] >= 0, 0)
            peak_ex.to_csv(f"reference/chipatlas/{srx}.ex.05.bed", index=False, header=False, sep="\t")
            peak = pybedtools.BedTool(f"reference/chipatlas/{srx}.ex.05.bed")
            peaks[tf] = peak
            tss_peak = tss.intersect(peak, wa=True)
            tss_peak = tss_peak.to_dataframe()
            attr = pybedtools.BedTool(f"attribution_seq/{study}/Norman.{tf}.test/00_all_Norman.{tf}_ixg_fc.bed")
            bed1 = attr
            bed2 = peak
            intersected = bed1.intersect(bed2, wa=True, f=64E-9)
            bed1 = bed1.to_dataframe()
            bed1.rename(columns={"name": "gene", "thickStart": "pert", "thickEnd":"attribution", "itemRgb": "peak"}, inplace=True)
            bed1["attribution"] = bed1["attribution"].abs()
            intersected = intersected.to_dataframe()
            intersected["chip_peak"] = 1
            intersected = intersected[intersected["name"].isin(list(tss_peak.name))]
            bed1 = bed1[bed1["gene"].isin(list(tss_peak.name))]
            eval_df = pd.merge(bed1, intersected.loc[:,["chrom", "start", "end", "chip_peak"]], on=["chrom", "start", "end"], how="left").fillna(0)
            eval_df.peak = eval_df.peak.abs()
            ovlap[tf] = eval_df
            per_stats = {}
            for gene in set(eval_df["gene"]):
                per_gene = eval_df.query('gene == @gene')
                if (per_gene["chip_peak"].sum() > 0) & (per_gene["attribution"].sum() > 0):
                    auprc = average_precision_score(per_gene["chip_peak"], per_gene["attribution"])
                    random = per_gene["chip_peak"].sum() / len(per_gene["chip_peak"])
                    auroc = roc_auc_score(per_gene["chip_peak"], per_gene["attribution"])
                    per_stats[gene] = {
                    "AUPRC" : auprc,
                    "AUROC" : auroc,
                    "AUPRC ratio (log2)" : np.log2(auprc / random)
                    }
                else:
                    continue
                peak_matrix = ovlap[tf].query('chip_peak == 1').groupby(["gene","pert"])["peak"].count().reset_index()
                res = pd.merge(pd.DataFrame(per_stats).T, peak_matrix, left_index=True, right_on="gene")
                res = pd.merge(res, df3[f"Norman.{tf}"], left_on="gene", right_index=True)
                res = pd.merge(res, df_corr.loc[:,["Gene", "Correlation"]], left_on="gene", right_on="Gene")
                stats[tf] = res
    summary_stats = pd.DataFrame()
    for tf in tfs:
        summary_stats = pd.concat([summary_stats, stats[tf]])
    summary_stats["pert"] = summary_stats["pert"].apply(lambda x : x.replace("Norman.", ""))
    os.makedirs(f'figures/{study}/attribution/chip_auprc', exist_ok=True)
    plot_barplot(summary_stats, study, pretrained_model, "AUPRC")
    plot_barplot(summary_stats, study, pretrained_model, "AUPRC ratio (log2)")









