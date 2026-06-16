import argparse
import os
import gc
import torch
import numpy as np
from genperturb.preprocess._genome import GenomeIntervalDataset
from genperturb.dataloaders._alphagenome_sequence import resolve_alphagenome_indices

os.environ["ALPHAGENOME_TORCH_BF16"] = "1"

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import BatchRMSNorm


MODEL_VERSIONS = ["all_folds", "fold_0", "fold_1", "fold_2", "fold_3"]
RANDOM_SEED = 0

FASTA_PATH = "fasta/GRCh38.p14.genome.fa"
BED_PATH   = "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed"
OUTPUT_DIR = "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHAGENOME_SEQ_LENGTH = 1_048_576   # 1 Mb (2^20)
CENTER_START = 4094
CENTER_END   = 4098                 # 4 bins = 512 bp
NUM_CENTER_BINS = CENTER_END - CENTER_START


def get_output_path(model_version: str) -> str:
    model_suffix = "" if model_version == "all_folds" else f"_{model_version}"
    return f"{OUTPUT_DIR}/alphagenome{model_suffix}_embedding.npy"


print(f"Loading dataset from {BED_PATH}...")
ds = GenomeIntervalDataset(
    bed_file=BED_PATH,
    fasta_file=FASTA_PATH,
    context_length=ALPHAGENOME_SEQ_LENGTH,
    return_seq_indices=True,
)
print(f"Dataset size: {len(ds)} intervals")

def extract_embeddings(model_version: str) -> None:
    print(f"Loading AlphaGenome model + official JAX weights (human), version={model_version}...")

    model = AlphaGenome()
    model.add_reference_heads("human")
    model.load_from_official_jax_model(model_version)

    model.to(DEVICE)
    model.eval()

    # Fix: BatchRMSNorm updates running_var even in eval mode due to a bug
    # in alphagenome_pytorch where update_running_var=True takes precedence
    # over self.training=False. This causes running stats to drift during
    # sequential inference, producing corrupted embeddings.
    for m in model.modules():
        if isinstance(m, BatchRMSNorm):
            m.update_running_var = False

    print(f"Model loaded on {DEVICE}")

    embeddings_128bp_list = []
    print(f"Starting embedding extraction for {model_version}...")

    with torch.inference_mode():
        for i in range(len(ds)):
            seq_indices = ds[i]  # (1,048,576,)
            seq = resolve_alphagenome_indices(
                seq_indices,
                sequence_key=ds.get_interval_key(i),
                seed=RANDOM_SEED,
            ).unsqueeze(0).to(DEVICE)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                embeddings_1bp, embeddings_128bp, embeddings_pair = model(
                    seq,
                    organism_index=0,     # human fixed
                    return_embeds=True
                )

            # Sanity check (detect NaN / Inf)
            if not torch.isfinite(embeddings_128bp).all().item():
                raise RuntimeError(
                    f"Non-finite values in embeddings_128bp at version={model_version}, i={i}"
                )

            # Center 4 bins: (1, 8192, C) -> (1, 4, C)
            center_emb = embeddings_128bp[:, CENTER_START:CENTER_END, :].detach().float().cpu().numpy()
            embeddings_128bp_list.append(center_emb)

            del seq, embeddings_1bp, embeddings_128bp, embeddings_pair
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if i % 100 == 0:
                print(f"[{model_version}: {i}/{len(ds)}] done", flush=True)

    # (N, 1, 4, C) -> (N, 4, C)
    arr_128bp = np.concatenate(embeddings_128bp_list, axis=0)
    output_path = get_output_path(model_version)
    np.save(output_path, arr_128bp)

    print(f"Saved embeddings to {output_path}")
    print(f"  - Center 4 bins (512bp) embeddings shape: {arr_128bp.shape}")
    print(f"  - Expected shape: (N, 4, C) where N={len(ds)} and C={arr_128bp.shape[-1]}")

    del model, embeddings_128bp_list, arr_128bp
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_version",
    action="append",
    choices=MODEL_VERSIONS,
    help="AlphaGenome checkpoint to extract. Repeat to select multiple versions. Default: all versions.",
)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model_versions = args.model_version or MODEL_VERSIONS
for model_version in model_versions:
    extract_embeddings(model_version)
