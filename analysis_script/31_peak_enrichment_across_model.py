# conda activate deeptools
import pandas as pd
import numpy as np
import pyBigWig
import pybedtools
import subprocess
import os


study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
dataname = "Norman"
study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'

fafai_path = f'fasta/GRCh38.p13.genome.fa.fai'
fafai = pd.read_csv(fafai_path, sep="\t", usecols=[0,1], names=["chrom", "length"])

segment = pybedtools.BedTool(f'fasta/chromosome_segment.bed')
segment = segment.sort(faidx=fafai_path)

workdir = f'across_study/deeptools/{dataname}'
os.makedirs(workdir, exist_ok=True)


def write_bigwig(segment_df, outpath, pretrained_model, pert):
    bw = pyBigWig.open(f"{outpath}/{pretrained_model}.{pert}_ixg_fc.bw", "w")
    header = [(chrom, length) for i,(chrom,length) in fafai.iterrows()]
    bw.addHeader(header, maxZooms=0)
    bw.addEntries(segment_df["chrom"].to_list(), segment_df["start"].to_list(),
        segment_df["end"].to_list(), values=segment_df["name"].to_list())
    bw.close()


pert = "Norman.CEBPA"

pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
]


for pretrained_model in pretrained_models:
    study = f'{study_name}__{pretrained_model}_transfer_epoch100_batch256_adamw5e3'
    bed = pybedtools.BedTool(f"attribution_seq//{study}/{pert}.test/00_all_{pert}_ixg_fc.bed")
    bed = bed.sort(faidx=fafai_path)
    segment_attr = segment.map(b=bed, c="8", o="absmax")
    segment_attr = segment_attr.sort(faidx=fafai_path)
    segment_df = segment_attr.to_dataframe()
    segment_df["name"][segment_df["name"] == "."] = "0"
    segment_df["name"] = segment_df["name"].astype("float")
    segment_df["name"] = segment_df["name"] * 100 / segment_df["name"].sum()
    write_bigwig(segment_df, workdir, pretrained_model, pert)


bigwigs = " ".join([f"{workdir}/{i}.{pert}_ixg_fc.bw" for i in pretrained_models])


samplelabels = " ".join(pretrained_models)
colors = ""

chips = ["SRX10184518.05"]
refbeds = " ".join([f"reference/chipatlas/{i}.bed" for i in chips])

matrix = f"{pert}_ixg_fc"

cmd1 = f'''
computeMatrix scale-regions -S {bigwigs} \
                            -R {refbeds} \
                            --beforeRegionStartLength 10000 \
                            --regionBodyLength 2000 \
                            --afterRegionStartLength 10000 \
                            --samplesLabel {samplelabels} \
                            --skipZeros -p 12 \
                            -o {workdir}/{matrix}.matrix.mat.gz
'''

subprocess.check_output(cmd1, shell=True)

cmd2 = f'''
plotProfile -m {workdir}/{matrix}.matrix.mat.gz \
              -out {workdir}/Enrichment.{matrix}.svg \
              --numPlotsPerRow 1 \
              --dpi 300 \
              --plotWidth 10 \
              --plotHeight 5 \
              --colors royalblue darkorange darkorange green green \
              --startLabel start \
              --endLabel end \
              --yMax 0.0035 \
              --yMin -0.001 \
              --plotTitle "ATAC-seq peaks enrichment"
'''

subprocess.check_output(cmd2, shell=True)

