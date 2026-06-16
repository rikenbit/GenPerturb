import os
from pathlib import Path

CWD = str(Path(__file__).resolve().parents[2])

STUDY_CONFIGS = {
    "Norman": {
        "study_key": "Norman",
        "seqlet_mode": "nonmatched",
        "study_name": "NormanWeissman2019_filtered_mixscape_exnp_train",
        "study_full": (
            "NormanWeissman2019_filtered_mixscape_exnp_train"
            "__alphagenome_transfer_epoch100_batch256_adamw5e3"
        ),
        "control_col": "Norman.NT",
        "pert_prefix": "Norman.",
        "nonmatched_sample_size": 8000,
    },
    "Martin": {
        "study_key": "Martin",
        "seqlet_mode": "nonmatched",
        "study_name": "MartinRufino2025_mixscape_exnp_train",
        "study_full": (
            "MartinRufino2025_mixscape_exnp_train"
            "__alphagenome_transfer_epoch100_batch256_adamw5e3"
        ),
        "control_col": "MartinRufino.NT",
        "pert_prefix": "MartinRufino.",
        "nonmatched_sample_size": 3000,
    },
    "Martin_full": {
        "study_key": "Martin",
        "seqlet_mode": "full",
        "study_name": "MartinRufino2025_mixscape_exnp_train",
        "study_full": (
            "MartinRufino2025_mixscape_exnp_train"
            "__alphagenome_transfer_epoch100_batch256_adamw5e3"
        ),
        "control_col": "MartinRufino.NT",
        "pert_prefix": "MartinRufino.",
        "nonmatched_sample_size": None,
    },
    "Norman_matched": {
        "study_key": "Norman",
        "seqlet_mode": "matched",
        "study_name": "NormanWeissman2019_filtered_mixscape_exnp_train",
        "study_full": (
            "NormanWeissman2019_filtered_mixscape_exnp_train"
            "__alphagenome_transfer_epoch100_batch256_adamw5e3"
        ),
        "control_col": "Norman.NT",
        "pert_prefix": "Norman.",
        "nonmatched_sample_size": None,
    },
    "Martin_matched": {
        "study_key": "Martin",
        "seqlet_mode": "matched",
        "study_name": "MartinRufino2025_mixscape_exnp_train",
        "study_full": (
            "MartinRufino2025_mixscape_exnp_train"
            "__alphagenome_transfer_epoch100_batch256_adamw5e3"
        ),
        "control_col": "MartinRufino.NT",
        "pert_prefix": "MartinRufino.",
        "nonmatched_sample_size": None,
    },
}

CONTEXT_LENGTH = 1_048_576
HALF_CONTEXT = (CONTEXT_LENGTH - 1) // 2
FASTA = "fasta/GRCh38.p14.genome.fa"

RANDOM_SEED = 20260424

DATA_DIR_TPL = "attribution_analysis/insilico_mutation/{study}"

FIG_DIR_TPL = "figures/{study_full}/insilico_mutation"

TABLE_DIR_TPL = "attribution_analysis/insilico_mutation/{study}/tables"
