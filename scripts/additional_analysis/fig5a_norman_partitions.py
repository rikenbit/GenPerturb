#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROGRAMS = {
    "Erythroid": ['CBL_CNN1', 'CBL_PTPN12', 'CBL_PTPN9', 'CBL_UBASH3B', 'SAMD1_PTPN12',
                  'SAMD1_UBASH3B', 'UBASH3B_CNN1', 'UBASH3B_PTPN12', 'UBASH3B_PTPN9',
                  'UBASH3B_UBASH3A', 'UBASH3B_ZBTB25', 'BPGM_SAMD1', 'PTPN1',
                  'PTPN12_PTPN9', 'PTPN12_UBASH3A', 'PTPN12_ZBTB25'],
    "Granulocyte": ['SPI1', 'CEBPA', 'CEBPB', 'CEBPE_CEBPA', 'CEBPE_RUNX1T1',
                    'CEBPE_SPI1', 'CEBPE', 'ETS2_CEBPE', 'KLF1_CEBPA', 'FOSB_CEBPE'],
    "Megakaryocyte": ['MAPK1_TGFBR2', 'MAPK1', 'ETS2_MAPK1', 'ETS2', 'CEBPB_MAPK1'],
    "Pioneer factor": ['FOXA1_FOXF1', 'FOXA1_FOXL2', 'FOXA1_HOXB9', 'FOXA3_FOXA1',
                       'FOXA3_FOXF1', 'FOXA3_FOXL2', 'FOXA3_HOXB9', 'FOXA3',
                       'FOXF1_FOXL2', 'FOXF1_HOXB9', 'FOXL2_MEIS1', 'HOXA13', 'HOXC13',
                       'POU3F2_FOXL2', 'TP73', 'MIDN', 'LYL1_IER5L', 'DUSP9_SNAI1',
                       'ZBTB10_SNAI1'],
    "Pro-growth": ['CEBPE_KLF1', 'KLF1', 'KLF1_BAK1', 'KLF1_TGFBR2', 'ELMSAN1',
                   'MAP2K3_SLC38A2', 'MAP2K3_ELMSAN1', 'MAP2K3', 'MAP2K3_MAP2K6',
                   'MAP2K6_ELMSAN1', 'MAP2K6', 'KLF1_MAP2K6'],
    "G1 cycle": ['CDKN1A', 'CDKN1B_CDKN1A', 'CDKN1B', 'CDKN1C_CDKN1A', 'CDKN1C'],
}
PROGRAM_COLORS = {
    "Erythroid": "#C0392B", "Granulocyte": "#3498DB", "Megakaryocyte": "#7E57C2",
    "Pioneer factor": "#F39C12", "Pro-growth": "#43A047", "G1 cycle": "#795548",
    "Other": "#BDBDBD",
}


def add_program_labels(ad):
    lookup = {pert: program for program, perts in PROGRAMS.items() for pert in perts}
    ad.obs["Perturbation"] = ad.obs_names.astype(str)
    ad.obs["Program"] = pd.Categorical(
        ad.obs.Perturbation.map(lookup).fillna("Other"),
        categories=[*PROGRAMS, "Other"], ordered=True,
    )


def plot_program(ad, kind, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    sc.pl.umap(ad, color="Program", palette=[PROGRAM_COLORS[x] for x in ad.obs.Program.cat.categories],
               size=28, ax=ax, show=False, title="Observed" if kind == "real" else "Fitted")
    ax.legend(title="Program", loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=7, title_fontsize=8, frameon=False, markerscale=.8)
    fig.savefig(out / f"{kind}_program.svg", bbox_inches="tight")
    fig.savefig(out / f"{kind}_program.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="AnnData, membership tables and metrics")
    p.add_argument("--figure-out", type=Path,
                   help="UMAP SVG/PNG destination; defaults to --out")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.figure_out is None:
        a.figure_out = a.out
    else:
        a.figure_out.mkdir(parents=True, exist_ok=True)
    study = "NormanWeissman2019_filtered_mixscape_exnp_train"
    full = study + "__alphagenome_transfer_epoch100_batch256_adamw5e3"
    df = pd.read_csv(a.root / "data" / (study + ".tsv"), sep="\t", index_col=0)
    pred = np.load(a.root / "prediction" / full / "prediction.npy")
    mask = df.training.eq("test").to_numpy()
    matrices = [df.drop(columns="training").to_numpy()[mask].T, pred[mask].T]
    ads = []
    for kind, matrix in zip(["real", "pred"], matrices):
        ad = anndata.AnnData(matrix, obs=pd.DataFrame(index=df.columns[1:]), var=pd.DataFrame(index=df.index[mask]))
        ad.var_names_make_unique()
        ad.layers["scaled"] = sc.pp.scale(ad, copy=True).X
        sc.tl.pca(ad, random_state=0)
        sc.pp.neighbors(ad, n_neighbors=10, random_state=0)
        sc.tl.leiden(ad, resolution=1.5, key_added="leiden", random_state=0)
        sc.tl.umap(ad, random_state=0)
        add_program_labels(ad)
        ad.write_h5ad(a.out / f"adata_{kind}.h5ad")
        ad.obs.to_csv(a.out / f"{kind}_membership.tsv", sep="\t")
        fig, ax = plt.subplots(figsize=(5, 4))
        sc.pl.umap(ad, color="leiden", ax=ax, show=False, title="Observed" if kind == "real" else "Fitted")
        fig.savefig(a.figure_out / f"{kind}_leiden.svg", bbox_inches="tight")
        fig.savefig(a.figure_out / f"{kind}_leiden.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        plot_program(ad, kind, a.figure_out)
        ads.append(ad)
    record = dict(n_conditions=ads[0].n_obs, n_loci=ads[0].n_vars, control_included=True,
                  ari=adjusted_rand_score(ads[0].obs.leiden, ads[1].obs.leiden),
                  nmi=normalized_mutual_info_score(ads[0].obs.leiden, ads[1].obs.leiden, average_method="arithmetic"),
                  analysis="matched-locus clustering", scanpy=sc.__version__, seed=0)
    (a.out / "partition_metrics.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
