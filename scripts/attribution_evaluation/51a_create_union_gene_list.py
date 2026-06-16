import numpy as np
import pandas as pd
import os
from pathlib import Path

CWD = str(Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(CWD, "attribution_analysis/captum_union")
os.makedirs(OUT_DIR, exist_ok=True)

STUDIES = {
    "norman": {
        "tsv": os.path.join(CWD, "data/NormanWeissman2019_filtered_mixscape_exnp_train.tsv"),
        "perturbations": [
            "Norman.IRF1", "Norman.TP73", "Norman.CEBPA", "Norman.HNF4A",
            "Norman.FOXA1", "Norman.AHR", "Norman.PRDM1", "Norman.SPI1",
            "Norman.SNAI1", "Norman.KMT2A", "Norman.CEBPB", "Norman.JUN",
            "Norman.ETS2", "Norman.EGR1",
        ],
        "control": "Norman.NT",
    },
    "martin": {
        "tsv": os.path.join(CWD, "data/MartinRufino2025_mixscape_exnp_train.tsv"),
        "perturbations": [
            "MartinRufino.BCL11A", "MartinRufino.FOSL1", "MartinRufino.GATA1",
            "MartinRufino.GATA2", "MartinRufino.GFI1B", "MartinRufino.KLF1",
            "MartinRufino.LDB1", "MartinRufino.LMO2", "MartinRufino.MYB",
            "MartinRufino.NFE2", "MartinRufino.RUNX1", "MartinRufino.SPI1",
            "MartinRufino.TAL1",
        ],
        "control": "MartinRufino.NT",
    },
}

N_TOP_GENES = 200


def create_union_gene_list(study_label, study_info):
    df = pd.read_csv(study_info["tsv"], sep="\t", index_col=0)
    value_cols = df.columns[1:]  # first column is 'training'
    df_val = df[value_cols]
    ctrl_col = study_info["control"]

    df_fc = (df_val.T - df_val[ctrl_col]).T.drop(columns=[ctrl_col])

    perturbations = study_info["perturbations"]
    union_genes = set()

    print(f"\n{'='*60}")
    print(f"{study_label.upper()}: {len(perturbations)} perturbations")
    print(f"{'='*60}")

    for pert in perturbations:
        top_genes = list(dict.fromkeys(
            df_fc.abs().sort_values(
                pert, ascending=False
            ).loc[:, pert].head(N_TOP_GENES * 2).index.tolist()
        ))[:N_TOP_GENES]

        union_genes.update(top_genes)
        print(f"  {pert}: top {len(top_genes)} genes (union so far: {len(union_genes)})")

    union_genes_sorted = sorted(union_genes)

    out_path = os.path.join(OUT_DIR, f"union_genes_{study_label}.txt")
    with open(out_path, "w") as f:
        for gene in union_genes_sorted:
            f.write(gene + "\n")

    print(f"\n  Saved: {out_path} ({len(union_genes_sorted)} genes)")

    pert_path = os.path.join(OUT_DIR, f"pert_list_{study_label}.txt")
    with open(pert_path, "w") as f:
        for pert in perturbations:
            f.write(pert + "\n")
    print(f"  Saved: {pert_path} ({len(perturbations)} perturbations)")

    return union_genes_sorted


def main():
    for study_label, study_info in STUDIES.items():
        create_union_gene_list(study_label, study_info)

    print("\nDone.")


if __name__ == "__main__":
    main()
