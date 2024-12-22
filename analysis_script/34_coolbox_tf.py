# conda activate coolbox
import pandas as pd
import numpy as np
import pyBigWig
import sys
import os
import coolbox
from coolbox.api import *
import seaborn as sns

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
    """
    For coolbox peak
    format : chr1    198620984       198621112       1       CRISPRa.ATL1    2.3338741e-06
    """
    attributions_bed6 = attributions_bed.loc[:,["chr", "start", "end", "gene", "pert", "attr"]]
    attributions_bed6.to_csv(output_bed6, sep="\t", header=False, index=False)

## common param ##
nbin = 128
attribution = "ixg"
context_length = 196_608

#study_name = sys.argv[1]
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
#study_suffix = sys.argv[2]
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'

tfs = ['HNF4A', 'TP73', 'IRF1', 'AHR', 'CEBPA', 'SPI1', 'KMT2A', 'PRDM1', 'CEBPB', 'SNAI1', 'FOXA1', 'ATAC', 'H3K27']
perts = [f"Norman.{i}" for i in tfs[:-2]]
srxs = ["SRX10475577", "DRX440350", "SRX2424509", "SRX4342285", "SRX097105", "SRX1431734", "SRX5732677", "SRX3070540", "SRX2423912", "SRX10478046", "SRX2424502", "SRX10184518", "SRX10184470"]

#gene = "PDCD5"
gene = "PPAN"
#gene = "PTPRH"
#gene = "CEBPE"




tss_path = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed"
tss = pd.read_csv(tss_path, sep="\t", usecols=[0,1,2,3], names=["chr", "start", "end", "gene"])
chrom, seq_start, seq_end = list(tss[tss["chr"].str.contains("chr")].query('gene == @gene').values[0])[:3]
seq_start  = int(seq_start - (context_length / 2))
seq_end    = int(seq_end   + (context_length / 2))
out_dir = f"coolbox/{study}/{gene}_{chrom}_{seq_start}_{seq_end}_{attribution}"
os.makedirs(out_dir, exist_ok=True)

attributions_path = f"attribution_seq/{study}/{gene}/00_all_{gene}_ixg_fc.bed"
attributions_bed_all  = pd.read_csv(attributions_path, sep="\t", names=["chr", "start", "end", "gene", "strand", "score", "pert", "attr", "peak"])
coolbox_attrs = []
coolbox_bed6s = []
max_values = np.array([])
min_values = np.array([])
for pert in perts:
    attributions_bed = attributions_bed_all.query('`pert` == @pert')
    attributions_mean = np.array(attributions_bed["attr"])
    bigwig = f"bigwig_{pert}.bw"
    write_bigwig(attributions_mean, chrom, seq_start, seq_end, nbin, f"{out_dir}/{bigwig}")
    coolbox_attrs.append(bigwig)
    bed6 = f"bed6_{pert}.bed"
    transform_bed6(attributions_bed[attributions_bed["peak"].isin([1, -1])], f"{out_dir}/{bed6}")
    coolbox_bed6s.append(bed6)
    max_values = np.append(max_values, np.quantile(attributions_mean, 0.99))
    min_values = np.append(max_values, np.quantile(attributions_mean, 0.00))


#### coolbox ####
path_bed  = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.unique.bed"
alabels = [i.replace("Norman.", "") for i in perts]
attrs = coolbox_attrs
attrpeaks = coolbox_bed6s

path_chip = "reference/chipatlas"
chips = []
chipspeaks = []
bw_list = {}
for i,j in zip(srxs, tfs):
    chips.append(f"{i}.bw")
    chipspeaks.append(f"{i}.05.bed6.bed")
    bw_list[j] = i


clabels = [next(key for key, value in bw_list.items() if peak.startswith(value)) for peak in chipspeaks]


view_start = seq_start
view_end   = seq_end
suffix = ""
max_value = max_values.max()
min_value = min_values.min()
cmap = sns.color_palette("Dark2") + sns.color_palette("Set1")

frame = Track()
plt.rcParams["font.size"] = 14
for attr_file, attrpeak_file, alabel, c in zip(attrs, attrpeaks, alabels, cmap[:len(attrs)]):
    frame += BigWig(f"{out_dir}/{attr_file}", max_value=max_value, min_value=min_value, color=c) + Title(f"{alabel}\n(Attribution)") + TrackHeight(2)


for chip_file, chippeak_file, clabel, c in zip(chips, chipspeaks, clabels, cmap[:len(chips)]):
    frame += BigWig(f"{path_chip}/{chip_file}", color=c) + Title(f"{clabel}\n(ChIP-seq)") + TrackHeight(2)
    frame += BED(f"{path_chip}/{chippeak_file}", labels=False, display="collapsed", color=c, border_color=None) + TrackHeight(0.4)


frame += XAxis()
frame += BED(path_bed, gene_style="flybase", color="black", border_color="black") + Title("Gene") + TrackHeight(6)
frame += FrameTitle(gene)
fig = Browser(frame, reference_genome='hg38')
fig.goto(f"{chrom}:{view_start}-{view_end}")
fig.save(f"{out_dir}/track_{gene}{suffix}.svg", dpi=300)

