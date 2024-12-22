# https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import glob
import os
from _genome import GenomeIntervalDataset
#from genperturb.dataloaders._genome import GenomeIntervalDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 160_000
output_name = "160k"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, trust_remote_code=True)



fasta = 'fasta/GRCh38.p13.genome.fa'
bed = 'fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed'

ds = GenomeIntervalDataset(
    bed_file = bed,
    fasta_file = fasta,
    return_sequence = True,
    context_length = max_length -2
)


model.to(device)
model.eval()

os.makedirs(f"data/hyena", exist_ok=True)

embeddings = []
tsss = []
lasts = []
means = []

with torch.inference_mode():
    for i in range(len(ds)):
        print(f"processing: {i} / {len(ds)}")
        sequence = ds[i]
        tok_seq = tokenizer(sequence)["input_ids"]
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)
        torch_outs = model(tok_seq, output_hidden_states=True)
        embedding = torch_outs['hidden_states'][-1].cpu().detach().numpy()[0]

        tsss.append(embedding[79743:80255])
        lasts.append(embedding[max_length-2])
        means.append(embedding.mean(axis=0))

        del embedding
        del tok_seq
        
        if i % 50 == 0:
            np_array_tss = np.stack([x for x in tsss])
            np.save(f'data/hyena/hyena_embedding_{output_name}_tss_{i // 50}.npy', np_array_tss)
            tsss = []
            del np_array_tss

            np_array_last = np.stack([x for x in lasts])
            np.save(f'data/hyena/hyena_embedding_{output_name}_last_{i // 50}.npy', np_array_last)
            lasts = []
            del np_array_last

            np_array_mean = np.stack([x for x in means])
            np.save(f'data/hyena/hyena_embedding_{output_name}_mean_{i // 50}.npy', np_array_mean)
            means = []
            del np_array_mean


np_array_tss = np.stack([x for x in tsss])
np.save(f'data/hyena/hyena_embedding_{output_name}_tss_{len(ds) // 50 + 1}.npy', np_array_tss)
np_array_last = np.stack([x for x in lasts])
np.save(f'data/hyena/hyena_embedding_{output_name}_last_{len(ds) // 50 + 1}.npy', np_array_last)
np_array_mean = np.stack([x for x in means])
np.save(f'data/hyena/hyena_embedding_{output_name}_mean_{len(ds) // 50 + 1}.npy', np_array_mean)



file_pattern = f'data/hyena/hyena_embedding_{output_name}_tss_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))
all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/hyena_embedding_{output_name}_tss.npy', all_embeddings)


file_pattern = f'data/hyena/hyena_embedding_{output_name}_last_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))
all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/hyena_embedding_{output_name}_last.npy', all_embeddings)



file_pattern = f'data/hyena/hyena_embedding_{output_name}_mean_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))
all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/hyena_embedding_{output_name}_mean.npy', all_embeddings)


