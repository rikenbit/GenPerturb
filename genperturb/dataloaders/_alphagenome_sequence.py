import hashlib
from typing import Any

import torch
import torch.nn.functional as F


ALPHAGENOME_RANDOM_SEED = 0


def alphagenome_sequence_key(chrom: str, start: int, end: int) -> str:
    """Return a stable key for deterministic ambiguous-base replacement."""
    return f"{chrom}:{int(start)}-{int(end)}"


def _seed_for_sequence(sequence_key: Any, seed: int) -> int:
    payload = f"{seed}\0{sequence_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63)


def resolve_alphagenome_indices(
    seq_indices: torch.Tensor,
    *,
    sequence_key: Any,
    seed: int = ALPHAGENOME_RANDOM_SEED,
) -> torch.Tensor:
    """Replace N=4 and padding=-1 with deterministic A/C/G/T indices.

    Replacement values depend only on ``sequence_key`` and ``seed``. They are
    therefore stable across extraction order, subsets, and process restarts.
    """
    seq = seq_indices.clone()
    ambiguous_mask = (seq < 0) | (seq > 3)
    n_ambiguous = int(ambiguous_mask.sum().item())
    if n_ambiguous == 0:
        return seq

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for_sequence(sequence_key, seed))
    replacements = torch.randint(
        0,
        4,
        (n_ambiguous,),
        dtype=seq.dtype,
        generator=generator,
        device="cpu",
    ).to(seq.device)
    seq[ambiguous_mask] = replacements
    return seq


def alphagenome_indices_to_one_hot(
    seq_indices: torch.Tensor,
    *,
    sequence_key: Any,
    seed: int = ALPHAGENOME_RANDOM_SEED,
) -> torch.Tensor:
    """Convert sequence indices to AlphaGenome-compatible differentiable one-hot."""
    resolved = resolve_alphagenome_indices(
        seq_indices,
        sequence_key=sequence_key,
        seed=seed,
    )
    return F.one_hot(resolved.long(), num_classes=4).float()
