#https://huggingface.co/EleutherAI/enformer-official-rough
import torch
from torch import nn
import polars as pl
from enformer_pytorch import Enformer, GenomeIntervalDataset
import pandas as pd
import numpy as np
import h5py
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_name = "masked"


fasta = 'fasta/GRCh38.p13.genome.fa'
bed = 'fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed'
context_length=196_608

short_context_length=10000
context_start= int(context_length / 2) - int(short_context_length / 2)
context_end  = int(context_length / 2) + int(short_context_length / 2)


ds = GenomeIntervalDataset(
    bed_file = bed,
    fasta_file = fasta,
    context_length = context_length
)

model = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=4)

model.to(device)
model.eval()
tensors = []
with torch.inference_mode():
    for i in range(len(ds)):
        seq = ds[i]
        seq[:context_start,:] = 0
        seq[context_end:,:] = 0
        tensor = model(seq.cuda(), return_only_embeddings = True)
        tensors.append(tensor.cpu().detach().numpy())
        print(i)

np_array = np.stack([x for x in tensors])
np.save(f"data/enformer_embedding_{output_name}.npy", np_array)



