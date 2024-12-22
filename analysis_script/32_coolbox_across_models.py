# conda activate coolbox
import pandas as pd
import numpy as np
import pyBigWig
import sys
import os
import coolbox
from coolbox.api import *
import seaborn as sns
import matplotlib.pyplot as plt

def write_bigwig(attributions_mean, chrom, seq_start, seq_end, nbin, output_bw):
    bw = pyBigWig.open(output_bw, "w")
    bw.addHeader([(chrom, 248956422)], maxZooms=0)
    chroms = np.array([chrom] * int(attributions_mean.shape[0]))
    bed = [i for i in range(seq_start, seq_end + 1, nbin)]
    starts = bed[:-1]
    ends = bed[1:] 
    bw.addEntries(chroms, starts, ends=ends, values=attributions_mean)
    bw.close()


def transform_bed6(attributions_bed, output_bed6):
    attributions_bed6 = attributions_bed.loc[:,["chr", "start", "end", "gene", "pert", "attr"]]
    attributions_bed6.to_csv(output_bed6, sep="\t", header=False, index=False)

## common param ##
nbin = 128
attribution = "ixg"
context_length = 196_608
pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
]


## study name ##
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
data_prefix = "Norman."

## gene and pert ##
gene = "TYROBP"
perts = ["Norman.CEBPA"]

## set genome position and outname ##
tss_path = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed"
tss = pd.read_csv(tss_path, sep="\t", usecols=[0,1,2,3], names=["chr", "start", "end", "gene"])
chrom, seq_start, seq_end = list(tss[tss["chr"].str.contains("chr")].query('gene == @gene').values[0])[:3]
seq_start  = int(seq_start - (context_length / 2))
seq_end    = int(seq_end   + (context_length / 2))
out_dir = f"across_study/coolbox/{study_name}/{gene}_{chrom}_{seq_start}_{seq_end}_{attribution}"
os.makedirs(out_dir, exist_ok=True)


## prepare bigwig/bed6 ##
model_results = {}
for pretrained_model in pretrained_models:
    study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
    study = f'{study_name}__{study_suffix}'
    attributions_path = f"attribution_seq/{study}/{perts[0]}.test/00_all_{perts[0]}_ixg_fc.bed"
    attributions_bed_all = pd.read_csv(attributions_path, sep="\t", names=["chr", "start", "end", "gene", "strand", "score", "pert", "attr", "peak"])
    coolbox_attrs = []
    coolbox_bed6s = []
    max_values = np.array([])
    min_values = np.array([])
    for pert in perts:
        attributions_bed = attributions_bed_all.query('pert == @pert & gene == @gene ')
        attributions_mean = np.array(attributions_bed["attr"])
        if pretrained_model in ["hyena_dna_tss", "hyena_dna_last"]:
            padding = int((1536 - 1250) / 2)
            attributions_mean = np.pad(attributions_mean, ((padding, padding)), 'constant')
        elif pretrained_model in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
            padding = int((1536 - 96) / 2)
            attributions_mean = np.pad(attributions_mean, ((padding, padding)), 'constant')
        bigwig = f"{pretrained_model}_{pert}.bw"
        write_bigwig(attributions_mean, chrom, seq_start, seq_end, nbin, f"{out_dir}/{bigwig}")
        coolbox_attrs.append(bigwig)
        bed6 = f"{pretrained_model}_{pert}.bed"
        transform_bed6(attributions_bed[attributions_bed["peak"].isin([1, -1])], f"{out_dir}/{bed6}")
        coolbox_bed6s.append(bed6)
        max_values = np.append(max_values, np.quantile(attributions_mean, 0.99))
        min_values = np.append(max_values, np.quantile(attributions_mean, 0.01))
    max_value = max_values.max()
    min_value = min_values.min()
    alabels = [i.replace(data_prefix, "") for i in perts]
    model_results[pretrained_model] = {
        "attrs": coolbox_attrs,
        "attrpeaks": coolbox_bed6s,
        "alabels": alabels,
        "max_values": max_value,
        "min_values": min_value
    }

## prepare chip-atlas ##
path_chip = "reference/chipatlas"
srxs = ["SRX10184470", "SRX2636149", "SRX8725106", "SRX9243952", "SRX028593", "SRX10184518"]
histons = ["H3K27ac", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3", "ATAC"]

chips = []
chippeaks = []
bw_list = {}
for i,j in zip(srxs, histons):
    chips.append(f"{i}.bw")
    chippeaks.append(f"{i}.05.bed6.bed")
    bw_list[j] = i


clabels = [next(key for key, value in bw_list.items() if peak.startswith(value)) for peak in chippeaks]


#### coolbox ####
name_replace = {
'enformer': 'Enformer',
'hyena_dna_tss': 'HyenaDNA TSS',
'hyena_dna_last': 'HyenaDNA last',
'nucleotide_transformer_tss': 'Nucleotide Transformer TSS',
'nucleotide_transformer_cls': 'Nucleotide Transformer CLS',
}

path_bed  = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.unique.bed"
view_start = seq_start
view_end   = seq_end
suffix = ""

cmap = sns.color_palette("deep")
cmap = [cmap[0]] + [cmap[1]] * 3 + [cmap[2]] * 3

frame = Track()
plt.rcParams["font.size"] = 16
# attribution #
for pretrained_model, c in zip(pretrained_models, cmap):
    attrs = model_results[pretrained_model]["attrs"]
    attrpeaks = model_results[pretrained_model]["attrpeaks"]
    alabels = model_results[pretrained_model]["alabels"]
    max_value = model_results[pretrained_model]["max_values"]
    min_value = model_results[pretrained_model]["min_values"]
    for attr_file, attrpeak_file, alabel in zip(attrs, attrpeaks, alabels):
        title = f'{alabel}\n({name_replace.get(pretrained_model, pretrained_model)})'
        frame += BigWig(f"{out_dir}/{attr_file}", max_value=max_value, min_value=min_value, color=c) + Title(title) + TrackHeight(2)

# Chip-atlas
cmap_chip = sns.color_palette("gist_stern_r")
for chip_file, chippeak_file, clabel, c in zip(chips, chippeaks, clabels, cmap_chip):
    frame += BigWig(f"{path_chip}/{chip_file}", color=c) + Title(f"{clabel}\n(ChIP-seq)") + TrackHeight(2)

frame += XAxis()
frame += BED(path_bed, gene_style="flybase", color="black", border_color="black", fontsize=17, row_height=0.2) + Title("Gene") + TrackHeight(5)
frame += FrameTitle(gene)
frame.properties["width"] = 50
frame.properties["margins"] = {'left': 0.1, 'right': 1, 'bottom': 0, 'top': 1}
frame.properties["width_ratios"] = (0.05, 0.6, 0.35)
fig = Browser(frame, reference_genome='hg38')
fig.goto(f"{chrom}:{view_start}-{view_end}")
fig.save(f"{out_dir}/track_{gene}{suffix}.svg", dpi=300)




