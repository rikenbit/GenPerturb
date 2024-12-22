#https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import re
import glob
import h5py
import os
from _genome import GenomeIntervalDataset
#from genperturb.dataloaders._genome import GenomeIntervalDataset


output_name = "v2_500m"
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

max_length = tokenizer.model_max_length
max_length_seq = ((max_length) * 6) - 6
#max_length_seq = ((max_length - 128) * 6)

ds = GenomeIntervalDataset(
    bed_file = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed",
    fasta_file = 'fasta/GRCh38.p13.genome.fa',
    return_sequence = True,
    context_length = max_length_seq,
)


model.cuda()
model.eval()

os.makedirs(f"data/nt", exist_ok=True)
f = open("data/nt_token_check.txt", 'a')

with torch.inference_mode():
    batch_size = 50
    batch_nums = list(range(len(ds) // batch_size + 1))
    for batch_num in batch_nums:
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ds))
        sequences = [ds[i] for i in range(start_idx, end_idx)]
        #tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length = max_length)["input_ids"]
        #attention_mask = tokens_ids != tokenizer.pad_token_id
        #torch_outs = model(tokens_ids.cuda(), attention_mask=attention_mask.cuda(), encoder_attention_mask=attention_mask.cuda(), output_hidden_states=True)
        tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", truncation=True)["input_ids"]
        torch_outs = model(tokens_ids.cuda(), output_hidden_states=True)
        embeddings = torch_outs['hidden_states'][-1].cpu().detach().numpy()
        lengthes = (tokens_ids != 1).sum(axis=1)
        for num in list(lengthes.numpy().astype("int")):
            print(str(num))
            f.write(str(num) + '\n')
        clss = embeddings[:,0,:]
        #tsss = embeddings[:,939:1111,:]
        tsss = embeddings[:,982:1068,:]
        means = embeddings.mean(-2)
    
        np.save(f'data/nt/nt_embedding_{output_name}_cls_{batch_num}.npy', clss)
        np.save(f'data/nt/nt_embedding_{output_name}_tss_{batch_num}.npy', tsss)
        np.save(f'data/nt/nt_embedding_{output_name}_mean_{batch_num}.npy', means)

        del sequences
        del tokens_ids
        del torch_outs
        del tsss

f.close()
## test ##

#batch_size = 3
#batch_num = 0
#start_idx = batch_num * batch_size
#end_idx = min((batch_num + 1) * batch_size, len(ds))
#sequences = [ds[i] for i in range(start_idx, end_idx)]
#tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length = max_length)["input_ids"]
#attention_mask = tokens_ids != tokenizer.pad_token_id
#torch_outs = model(tokens_ids.cuda(), attention_mask=attention_mask.cuda(), encoder_attention_mask=attention_mask.cuda(), output_hidden_states=True)
#embeddings = torch_outs['hidden_states'][-1].cpu().detach().numpy()

########



file_pattern = f'data/nt/nt_embedding_{output_name}_cls_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))

all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/nt_embedding_{output_name}_cls.npy', all_embeddings)



file_pattern = f'data/nt/nt_embedding_{output_name}_tss_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))

all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/nt_embedding_{output_name}_tss.npy', all_embeddings)




file_pattern = f'data/nt/nt_embedding_{output_name}_mean_{{}}.npy'
last_file_num = len(glob.glob(file_pattern.format('*')))

all_embeddings = []
for i in range(last_file_num):
    embeddings = np.load(file_pattern.format(i))
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings)
np.save(f'data/nt_embedding_{output_name}_mean.npy', all_embeddings)



