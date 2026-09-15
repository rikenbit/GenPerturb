#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

STUDIES = ["NormanWeissman2019_filtered_mixscape_exnp_train",
           "ReplogleWeissman2022_K562_essential_mixscape_exnp_train", "ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
           "ReplogleWeissman2022_rpe1_mixscape_exnp_train", "JialongJiang2024_Myeloid_train",
           "JialongJiang2024_CD4T_train", "JialongJiang2024_CD8T_train", "JialongJiang2024_B_cell_train",
           "Srivatsan2019_A549_train", "Srivatsan2019_K562_train", "Srivatsan2019_MCF7_train",
           "MartinRufino2025_mixscape_exnp_train"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    a = p.parse_args()
    configs = [(s, m) for s in STUDIES for m in ["alphagenome", "borzoi", "enformer", "simplecnn"]]
    configs += [(s, f"alphagenome_fold_{f}") for s in [STUDIES[0], STUDIES[6]] for f in range(4)]
    study, model = configs[a.index]
    out = a.out / f"run_{a.index:02d}"
    out.mkdir(parents=True, exist_ok=False)
    fold = model.replace("alphagenome_", "") if "fold_" in model else "all_folds" if model == "alphagenome" else "primary"
    suffix = model + ("_baseline_epoch150_batch2_adamw5e3" if model == "simplecnn" else "_transfer_epoch100_batch256_adamw5e3")
    full = study + "__" + suffix
    obs_name = study + ("_" + model if "fold_" in model else "")
    observed = a.root / "data" / (obs_name + ".tsv")
    bed = a.root / "fasta" / (obs_name + ".bed")
    pred = a.root / "prediction" / full / "prediction.npy"
    evidence = dict(study=full, observed=str(observed), bed=str(bed), prediction=str(pred),
                    row_order_evidence="pipeline/02_slurm_script.py and GenPerturb.impute preserve input order; BED/TSV sequence checked")
    (out / "input_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not observed.exists() or not bed.exists() or not pred.exists():
        raise FileNotFoundError(f"Missing saved input for {full}")
    header = pd.read_csv(observed, sep="\t", nrows=0).columns.tolist()
    control = header[2]
    if not any(x in control.lower() for x in ["control", "nt", "dmso", "non-targeting", "vehicle"]):
        raise ValueError(f"Unrecognized first expression control channel: {control}")
    row = dict(study=study, backbone=model.split("_fold_")[0], fold=fold,
               observed=str(observed), prediction=str(pred), control=control, bed=str(bed))
    pd.DataFrame([row]).to_csv(out / "manifest.tsv", sep="\t", index=False)
    subprocess.run([sys.executable, str(Path(__file__).with_name("fig3_figS5_expression_summary.py")),
                    "--manifest", str(out / "manifest.tsv"), "--out", str(out / "output")], check=True)


if __name__ == "__main__":
    main()
