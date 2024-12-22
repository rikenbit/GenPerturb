# conda activate deeptools
import pandas as pd 
import numpy as np
import pyBigWig
import pybedtools
import os



def split_into_segments_vectorized(df):
    chromosomes = df['chrom'].to_numpy()
    lengths = df['length'].to_numpy()
    split_segments_df = pd.DataFrame(columns=['chrom', 'start', 'end'])
    for chrom, length in zip(chromosomes, lengths):
        segment_starts = np.arange(0, length.sum(), 128)  # Generate segment start positions for all chromosomes
        segment_ends = segment_starts + 128  # Calculate segment end positions
        segment_ends = np.where(segment_ends > length, length, segment_ends)  # Adjust end positions for last segment
        split_chrom_df = pd.DataFrame({'chrom': np.repeat(chrom, segment_starts.shape[0]),
                                          'start': segment_starts,
                                          'end': segment_ends})
        split_segments_df = pd.concat([split_segments_df, split_chrom_df])
    return split_segments_df

fafai_path = f'fasta/GRCh38.p13.genome.fa.fai'
fafai = pd.read_csv(fafai_path, sep="\t", usecols=[0,1], names=["chrom", "length"])


split_segments_df = split_into_segments_vectorized(fafai)
split_segments_df.to_csv(f'fasta/chromosome_segment.bed', sep="\t", index=False, header=False)

