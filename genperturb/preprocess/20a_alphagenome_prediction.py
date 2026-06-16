import os
import gc
import torch
import numpy as np
from genperturb.preprocess._genome import GenomeIntervalDataset
from genperturb.dataloaders._alphagenome_sequence import resolve_alphagenome_indices

os.environ["ALPHAGENOME_TORCH_BF16"] = "1"

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import BatchRMSNorm


MODEL_VERSION = "all_folds"
RANDOM_SEED = 0


FASTA_PATH = "fasta/GRCh38.p14.genome.fa"
BED_PATH   = "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed"
OUTPUT_DIR = "data"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHAGENOME_SEQ_LENGTH = 1_048_576   # 1 Mb (2^20)
CENTER_START = 4094
CENTER_END   = 4098                 # 4 bins = 512 bp
NUM_CENTER_BINS = CENTER_END - CENTER_START

# Genome track heads from add_reference_heads (order matches jax_genome_track_heads)
GENOME_TRACK_HEADS = ['rna_seq', 'cage', 'dnase', 'procap', 'atac', 'chip_tf', 'chip_histone']


print(f"Loading dataset from {BED_PATH}...")
ds = GenomeIntervalDataset(
    bed_file=BED_PATH,
    fasta_file=FASTA_PATH,
    context_length=ALPHAGENOME_SEQ_LENGTH,
    return_seq_indices=True,
)
print(f"Dataset size: {len(ds)} intervals")


print(f"Loading AlphaGenome model + official JAX weights (human), version={MODEL_VERSION}...")

model = AlphaGenome()
model.add_reference_heads("human")
model.load_from_official_jax_model(MODEL_VERSION)

model.to(DEVICE)
model.eval()

# Fix: BatchRMSNorm updates running_var even in eval mode due to a bug
# in alphagenome_pytorch where update_running_var=True takes precedence
# over self.training=False. This causes running stats to drift during
# sequential inference, producing corrupted predictions.
for m in model.modules():
    if isinstance(m, BatchRMSNorm):
        m.update_running_var = False

print(f"Model loaded on {DEVICE}")


predictions_list = []

print("Starting prediction extraction...")

with torch.inference_mode():
    for i in range(len(ds)):
        seq_indices = ds[i]  # (1,048,576,)
        seq = resolve_alphagenome_indices(
            seq_indices,
            sequence_key=ds.get_interval_key(i),
            seed=RANDOM_SEED,
        ).unsqueeze(0).to(DEVICE)

        # Forward: get track predictions (not embeddings)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(seq, organism_index=0)

        # Collect 128bp predictions from all genome track heads and concatenate
        # Head order: rna_seq(768), cage(640), dnase(384), procap(128), atac(256), chip_tf(1664), chip_histone(1152)
        tracks_128bp_parts = []
        for head_name in GENOME_TRACK_HEADS:
            head_out = out['human'][head_name]
            tracks_128bp_parts.append(head_out['scaled_predictions_128bp'])
        tracks_128bp = torch.cat(tracks_128bp_parts, dim=-1)  # (1, 8192, total_tracks)

        # Sanity check (detect NaN / Inf)
        if not torch.isfinite(tracks_128bp).all().item():
            raise RuntimeError(f"Non-finite values in 128bp_tracks at i={i}")

        # Center 4 bins: (1, 8192, n_tracks) -> (1, 4, n_tracks)
        center_pred = tracks_128bp[:, CENTER_START:CENTER_END, :].detach().float().cpu().numpy()
        predictions_list.append(center_pred)

        del seq, out, tracks_128bp, tracks_128bp_parts
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if i % 100 == 0:
            print(f"[{i}/{len(ds)}] done", flush=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = f"{OUTPUT_DIR}/alphagenome_prediction.npy"

# (N, 1, 4, T) -> (N, 4, T)
arr = np.concatenate(predictions_list, axis=0)
np.save(output_path, arr)

print(f"Saved predictions to {output_path}")
print(f"  - Center 4 bins (512bp) predictions shape: {arr.shape}")
print(f"  - Expected shape: (N, 4, T) where N={len(ds)} and T={arr.shape[-1]}")
