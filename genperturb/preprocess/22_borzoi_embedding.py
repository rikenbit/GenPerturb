import argparse, os, gc
import torch
import numpy as np
import re
from huggingface_hub import hf_hub_download
from borzoi_pytorch import Borzoi
from genperturb.preprocess._genome import GenomeIntervalDataset



parser = argparse.ArgumentParser(
    description="Extract Borzoi embeddings and save to .npy"
)
parser.add_argument(
    "--context_length",
    type=int,
    default=524_288,
    help="Context length of intervals (bp). "
         "If not 524,288 the sequence is zero-padded to that length."
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="johahi/borzoi-replicate-0", 
    help="Hugging Face repo ID of the Borzoi checkpoint"
)
args = parser.parse_args()
ctx_len = args.context_length
repo_id = args.repo_id

device         = "cuda" if torch.cuda.is_available() else "cpu"
emb            = 16
fasta          = "fasta/GRCh38.p14.genome.fa"
bed            = "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed"
output_tag     = f"{ctx_len//1000}k"


def pad_to_524288(seq_1hot: torch.Tensor) -> torch.Tensor:
    """Pad (L,4) one-hot tensor to (524288,4) with zeros, equal on both sides."""
    L = seq_1hot.shape[0]
    if L == 524_288:
        return seq_1hot
    if L > 524_288:
        raise ValueError(f"Input length {L} > 524 288; truncation not allowed.")
    pad_total  = 524_288 - L
    pad_left   = pad_total // 2
    pad_right  = pad_total - pad_left
    padded = torch.zeros((524_288, 4), dtype=seq_1hot.dtype)
    padded[pad_left:pad_left + L] = seq_1hot
    return padded


ds = GenomeIntervalDataset(
    bed_file       = bed,
    fasta_file     = fasta,
    context_length = ctx_len,
)

model = Borzoi.from_pretrained(
    repo_id,
    return_center_bins_only=True,
    bins_to_return=emb
)


state_path  = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
state_dict  = torch.load(state_path, map_location="cpu", weights_only=True)
model.to_empty(device=device)
model.load_state_dict(state_dict, strict=True)
model.eval()


z_crop_list, z_final_list = [], []
with torch.inference_mode():
    for i in range(len(ds)):
        seq = ds[i]                                         # (L,4)
        if ctx_len != 524_288:
            seq = pad_to_524288(seq)                        # (524288,4)

        seq = seq.permute(1, 0).unsqueeze(0).to(device)     # (1,4,L)

        z_crop  = model.get_embs_after_crop(seq)            # (1, emb, •)
        z_final = model.final_joined_convs(z_crop)          # (1, emb, •)

        z_crop_list.append(z_crop.cpu().numpy())
        z_final_list.append(z_final.cpu().numpy())

        del seq, z_crop, z_final
        torch.cuda.empty_cache()
        gc.collect()

        if i % 100 == 0:
            print(f"[{i}/{len(ds)}] done", flush=True)


os.makedirs("data", exist_ok=True)

match = re.search(r"/(.*?)(?:[-_]|$)", repo_id)
model_name = match.group(1) if match else "unknown"

np.save(f"data/{model_name}_embedding.npy", np.stack(z_final_list))


