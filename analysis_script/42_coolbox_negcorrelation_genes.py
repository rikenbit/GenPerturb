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

## prepare hi-c ##
path_hic = "reference/4DN/Cell2014"
if not os.path.exists(f"{path_hic}/4DNFI4EFYN3Q.bed6.bed"):
    tadb = pd.read_csv(f"{path_hic}/4DNFI4EFYN3Q.bed", sep="\t", header=None)
    tadb[5] = "."
    tadb.to_csv(f"{path_hic}/4DNFI4EFYN3Q.bed6.bed", sep="\t", header=False, index=False)


def write_bigwig(attr, outpath, attr_gene):
    fafai_path = f'fasta/GRCh38.p13.genome.fa.fai'
    fafai = pd.read_csv(fafai_path, sep="\t", usecols=[0,1], names=["chrom", "length"])
    bw = pyBigWig.open(f"figures/{study}/neg_cor/attr_{attr_gene}.bw", "w")
    header = [(chrom, length) for i,(chrom,length) in fafai.iterrows()]
    bw.addHeader(header, maxZooms=0)
    bw.addEntries(attr[0].to_list(), attr[1].to_list(), attr[2].to_list(), values=attr["abs_attr"].to_list())
    bw.close() 


def plot_track():
    view_start = seq_start
    view_end   = seq_end
    suffix = ""
    frame = Track()
    plt.rcParams["font.size"] = 14
    ## Hi-C ##
    cmap = sns.color_palette("deep")
    frame += Cool(f"{path_hic}/4DNFI18UHVRO.mcool", cmap="YlOrRd", style='triangular', color_bar='vertical') + Title("Hi-C\ncontact map")
    frame += BigWig(f"{path_hic}/4DNFIXU7QLG6.bw", color="darkred", style="line") + Title(f"Insulation\nscore")
    frame += BED(f"{path_hic}/4DNFI4EFYN3Q.bed6.bed", labels=False, display="collapsed", color="darkred", border_color=None) + Title(f"TAD\nboundaries") + TrackHeight(1)
    frame += BigWig(f"{path_chip}/ERX989298.bw", color="darkred") + Title(f"CTCF\n(ChIP-Atlas)") + TrackHeight(2)
    ## Attribution ##
    cmap_attr = sns.color_palette("Dark2")
    for attr_file, alabel, c in zip(attrs, alabels, cmap_attr):
        frame += BigWig(f"{attr_file}", color=c) + Title(alabel) + TrackHeight(2)
    ## ChIP-seq ##
    cmap_chip = sns.color_palette("gist_stern_r")
    for chip_file, chippeak_file, clabel, c in zip(chips, chippeaks, clabels, cmap_chip):
        frame += BigWig(f"{path_chip}/{chip_file}", color=c) + Title(f"{clabel}\n(ChIP-seq)") + TrackHeight(2)
    ## Track ##
    frame += XAxis()
    frame += BED(path_bed, gene_style="flybase", color="black", border_color="black") + Title("Gene") + TrackHeight(6)
    frame += XAxis()
    frame += FrameTitle(gene)
    fig = Browser(frame, reference_genome='hg38')
    fig.goto(f"{chrom}:{view_start}-{view_end}")
    fig.save(f'figures/{study}/neg_cor/track_{gene}.svg', dpi=300)


## set genome region ##
gene = "NFKBIB"
attr_genes = ['RINL', 'NFKBIB', 'SIRT2', 'MRPS12', 'FBXO17']
#gene = "ZNF581"
#attr_genes = ['ZNF579', 'ZNF524', 'ZNF581', 'CCDC106', 'U2AF2', 'EPN1'] 
context_length = 196_608
path_bed  = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.unique.bed"
tss_path = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed"
tss = pd.read_csv(tss_path, sep="\t", usecols=[0,1,2,3], names=["chr", "start", "end", "gene"])
chrom, seq_start, seq_end = list(tss[tss["chr"].str.contains("chr")].query('gene == @gene').values[0])[:3]
seq_start  = int(seq_start - (context_length / 2))
seq_end    = int(seq_end   + (context_length / 2))



## bigwig and coolbox ##
pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
]

for pretrained_model in pretrained_models:
    study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
    study_suffix = f"{pretrained_model}_transfer_epoch100_batch256_adamw5e3"
    study = f'{study_name}__{study_suffix}'
    os.makedirs(f'figures/{study}/neg_cor', exist_ok=True)
    ## prepare attribution bigwig ##
    for attr_gene in attr_genes:
        attr = pd.read_csv(f"attribution_seq/{study}/{attr_gene}/00_all_{attr_gene}_ixg_fc.bed", sep="\t", header=None)
        attr["abs_attr"] = attr[7].abs()
        attr_max = attr.groupby([0,1,2])[["abs_attr"]].max().reset_index()
        write_bigwig(attr_max, f'figures/{study}/neg_cor', attr_gene)
    attrs = [f'figures/{study}/neg_cor/attr_{attr_gene}.bw' for attr_gene in attr_genes]
    alabels = [f'{attr_gene}\n(Attribution)' for attr_gene in attr_genes]
    plot_track()


