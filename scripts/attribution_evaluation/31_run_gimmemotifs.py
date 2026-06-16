#!/usr/bin/env python

import os
import sys
import subprocess
import pandas as pd


if len(sys.argv) >= 3:
    study_name   = sys.argv[1]
    study_suffix = sys.argv[2]
else:
    study_name   = "NormanWeissman2019_filtered_mixscape_exnp_train"
    study_suffix = "alphagenome_transfer_epoch100_batch256_adamw5e3"

study = f"{study_name}__{study_suffix}"

single_tf = sys.argv[3] if len(sys.argv) >= 4 else None

genome = "fasta/GRCh38.p14.genome"
known_db = "reference/gimmemotifs/motif_db/JASPAR2022_vertebrates.pfm"

tf_list = pd.read_csv(
    "reference/humantfs/DatabaseExtract_v_1.01.txt",
    sep="\t",
    usecols=["HGNC symbol"]
)["HGNC symbol"].to_list()

if single_tf:
    tfs = [single_tf]
    print(f"[INFO] Single TF mode: processing {single_tf}")
else:
    cre_dir = f"cre/{study}/"
    tfs = [i for i in os.listdir(cre_dir) if any(j in i for j in tf_list)]

bed_configs = [
    {
        "name": "attribution_re2g_extended",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_re2g_extended_{{pert}}.bed",
        "output_suffix": "attribution_re2g_extended",
    },
    {
        "name": "attribution_re2g_extended_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_re2g_extended_{{pert}}.shuffle.bed",
        "output_suffix": "attribution_re2g_extended_shuffle",
    },
    {
        "name": "attribution_re2g",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_re2g_{{pert}}.bed",
        "output_suffix": "attribution_re2g",
    },
    {
        "name": "attribution_re2g_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_re2g_{{pert}}.shuffle.bed",
        "output_suffix": "attribution_re2g_shuffle",
    },
    {
        "name": "attribution_abc",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_abc_{{pert}}.bed",
        "output_suffix": "attribution_abc",
    },
    {
        "name": "attribution_abc_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_abc_{{pert}}.shuffle.bed",
        "output_suffix": "attribution_abc_shuffle",
    },
    {
        "name": "attribution",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_{{pert}}.bed",
        "output_suffix": "attribution",
    },
    {
        "name": "attribution_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/attribution_{{pert}}.shuffle.bed",
        "output_suffix": "attribution_shuffle",
    },
    {
        "name": "re2g_extended",
        "bed_path_tpl": f"cre/{study}/{{pert}}/re2g_extended_{{pert}}.bed",
        "output_suffix": "re2g_extended",
    },
    {
        "name": "re2g_extended_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/re2g_extended_{{pert}}.shuffle.bed",
        "output_suffix": "re2g_extended_shuffle",
    },
    {
        "name": "re2g",
        "bed_path_tpl": f"cre/{study}/{{pert}}/re2g_{{pert}}.bed",
        "output_suffix": "re2g",
    },
    {
        "name": "re2g_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/re2g_{{pert}}.shuffle.bed",
        "output_suffix": "re2g_shuffle",
    },
    {
        "name": "abc_score",
        "bed_path_tpl": f"cre/{study}/{{pert}}/abc_score_{{pert}}.bed",
        "output_suffix": "abc_score",
    },
    {
        "name": "abc_score_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/abc_score_{{pert}}.shuffle.bed",
        "output_suffix": "abc_score_shuffle",
    },
    {
        "name": "fanta_bio",
        "bed_path_tpl": f"cre/{study}/{{pert}}/fanta_bio_{{pert}}.bed",
        "output_suffix": "fanta_bio",
    },
    {
        "name": "fanta_bio_shuffle",
        "bed_path_tpl": f"cre/{study}/{{pert}}/fanta_bio_{{pert}}.shuffle.bed",
        "output_suffix": "fanta_bio_shuffle",
    },
    {
        "name": "tss_1kbp",
        "bed_path_tpl": f"cre/{study}/{{pert}}/tss_1kbp_{{pert}}.bed",
        "output_suffix": "tss_1kbp",
    },
    {
        "name": "chip_seq",
        "bed_path_tpl": f"cre/{study}/{{pert}}/chip_seq_{{tf_symbol}}.bed",
        "output_suffix": "chip_seq",
        "use_tf_symbol": True,
    },
]

for pert in tfs:
    tf_symbol = pert.split(".")[-1]
    
    for cfg in bed_configs:
        bed_name = cfg["name"]
        output_suffix = cfg["output_suffix"]
        
        if cfg.get("use_tf_symbol", False):
            bed_path = cfg["bed_path_tpl"].format(pert=pert, tf_symbol=tf_symbol)
        else:
            bed_path = cfg["bed_path_tpl"].format(pert=pert)

        if not os.path.exists(bed_path):
            print(f"[WARN] Missing bed: {bed_path} (skip {pert}/{bed_name})")
            continue

        gimme_outdir = f"gimme_results/{study}/{pert}/{output_suffix}"
        os.makedirs(gimme_outdir, exist_ok=True)

        cmd = [
            "gimme", "motifs",
            bed_path,
            gimme_outdir,
            "-g", genome,
            "-N", str(16),
            "--known",
            "-p", known_db,
        ]

        print(f"\n[INFO] Running: {pert} / {bed_name}")
        print(f"[INFO] CMD: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)