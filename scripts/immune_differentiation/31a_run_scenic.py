#!/usr/bin/env python3
import os
import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from arboreto.algo import grnboost2
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
ADATA_PATH = os.path.join(CWD, "data/adata/NormanWeissman2019_filtered_mixscape_exnp.h5ad")
TF_LIST = os.path.join(CWD, "reference/pyscenic/allTFs_hg38.txt")
RANKING_DB = os.path.join(CWD, "reference/pyscenic/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
MOTIF_ANNO = os.path.join(CWD, "reference/pyscenic/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")

STUDY_FULL = "NormanWeissman2019_filtered_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
OUTDIR = os.path.join(CWD, f"figures/{STUDY_FULL}/master_regulator/scenic")
os.makedirs(OUTDIR, exist_ok=True)

NUM_WORKERS = 32


def main():
    t0 = time.time()

    print("[1/6] Loading scRNA-seq data...")
    if not os.path.exists(ADATA_PATH):
        raw_path = os.path.join(CWD, "data/adata/NormanWeissman2019_filtered.h5ad")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Neither {ADATA_PATH} nor {raw_path} found")
        print("  Creating mixscape_exnp from raw download...")
        preprocess_norman(raw_path, ADATA_PATH)

    adata = sc.read_h5ad(ADATA_PATH)
    sc.pp.filter_genes(adata, min_cells=int(adata.n_obs * 0.01))  # >= 1% of cells
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    tf_names_all = set(line.strip() for line in open(TF_LIST) if line.strip())
    tf_in_data = [g for g in adata.var_names if g in tf_names_all]
    extra_tfs = [g for g in tf_in_data if g not in adata_hvg.var_names]
    if extra_tfs:
        adata_extra = adata[:, extra_tfs].copy()
        import anndata
        adata_hvg = anndata.concat([adata_hvg, adata_extra], axis=1)
        print(f"  Added {len(extra_tfs)} extra TFs → total: {adata_hvg.n_vars}")

    if hasattr(adata_hvg.X, 'toarray'):
        ex_matrix = pd.DataFrame(
            adata_hvg.X.toarray(),
            index=adata_hvg.obs_names,
            columns=adata_hvg.var_names,
        )
    else:
        ex_matrix = pd.DataFrame(
            adata_hvg.X,
            index=adata_hvg.obs_names,
            columns=adata_hvg.var_names,
        )
    print(f"  Expression matrix: {ex_matrix.shape}")

    print("[2/6] Loading TF list and ranking database...")
    tf_names_in_data = [tf for tf in tf_names_all if tf in ex_matrix.columns]
    print(f"  TFs in reference: {len(tf_names_all)}, in data: {len(tf_names_in_data)}")

    db = RankingDatabase(RANKING_DB, name="hg38_10kbp")
    print(f"  Ranking database loaded: {db.name}")

    adj_path = os.path.join(OUTDIR, "adjacencies.tsv")
    if os.path.exists(adj_path):
        print("[3/6] Loading cached GRNBoost2 adjacencies...")
        adjacencies = pd.read_csv(adj_path, sep="\t")
        print(f"  Loaded: {len(adjacencies)} edges")
    else:
        print("[3/6] Running GRNBoost2 (network inference)...")
        t1 = time.time()
        adjacencies = grnboost2(
            ex_matrix,
            tf_names=tf_names_in_data,
            verbose=True,
            seed=42,
        )
        print(f"  GRNBoost2 done: {len(adjacencies)} edges ({time.time()-t1:.1f}s)")
        adjacencies.to_csv(adj_path, sep="\t", index=False)
        print(f"  Saved: {adj_path}")

    print("[4/6] Running cisTarget (regulon prediction)...")
    t2 = time.time()

    modules = list(modules_from_adjacencies(adjacencies, ex_matrix))
    print(f"  Modules: {len(modules)}")

    df_motifs = prune2df(
        [db],
        modules,
        MOTIF_ANNO,
        num_workers=NUM_WORKERS,
    )
    print(f"  Motif enrichment done: {len(df_motifs)} entries ({time.time()-t2:.1f}s)")

    motif_path = os.path.join(OUTDIR, "motif_enrichment.csv")
    df_motifs.to_csv(motif_path)
    print(f"  Saved: {motif_path}")

    df_motifs = pd.read_csv(motif_path, index_col=[0, 1], header=[0, 1])
    df_motifs[("Enrichment", "TargetGenes")] = df_motifs[("Enrichment", "TargetGenes")].apply(
        lambda x: eval(x, {"np": np, "__builtins__": {}}) if isinstance(x, str) else x
    )

    regulons = df2regulons(df_motifs)
    print(f"  Regulons: {len(regulons)}")

    regulon_path = os.path.join(OUTDIR, "regulons.pkl")
    with open(regulon_path, "wb") as f:
        pickle.dump(regulons, f)
    print(f"  Saved: {regulon_path}")

    for reg in sorted(regulons, key=lambda r: -len(r.genes))[:20]:
        print(f"    {reg.name}: {len(reg.genes)} target genes")

    print("[5/6] Running AUCell (regulon activity scoring)...")
    t3 = time.time()

    auc_mtx = aucell(
        ex_matrix,
        regulons,
        num_workers=NUM_WORKERS,
    )
    print(f"  AUCell done: {auc_mtx.shape} ({time.time()-t3:.1f}s)")

    aucell_path = os.path.join(OUTDIR, "aucell_scores.tsv")
    auc_mtx.to_csv(aucell_path, sep="\t")
    print(f"  Saved: {aucell_path}")

    print("[6/6] Aggregating AUCell scores per perturbation...")

    pert_col = None
    for col_name in ["gene", "perturbation", "condition"]:
        if col_name in adata.obs.columns:
            pert_col = col_name
            break
    if pert_col is None:
        raise ValueError(f"Cannot find perturbation column in obs: {list(adata.obs.columns)}")

    pert_labels = adata.obs[pert_col].values
    print(f"  Using perturbation column: '{pert_col}', unique: {len(set(pert_labels))}")
    auc_mtx["perturbation"] = pert_labels

    auc_mean = auc_mtx.groupby("perturbation").mean()
    auc_mean_path = os.path.join(OUTDIR, "aucell_mean_per_perturbation.tsv")
    auc_mean.to_csv(auc_mean_path, sep="\t")
    print(f"  Perturbations: {len(auc_mean)}")
    print(f"  Saved: {auc_mean_path}")

    regulon_names_path = os.path.join(OUTDIR, "regulon_names.txt")
    with open(regulon_names_path, "w") as f:
        for reg in regulons:
            n_genes = len(reg.genes)
            f.write(f"{reg.name}\t{n_genes}\n")
    print(f"  Saved: {regulon_names_path}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def preprocess_norman(raw_path, out_path):
    import pertpy as pt
    import scipy as sp
    import gc

    print(f"  Loading raw data: {raw_path}")
    adata = sc.read_h5ad(raw_path)
    adata.obs.rename(columns={"perturbation": "gene"}, inplace=True)
    adata.var.rename(columns={"ensemble_id": "ensembl_id"}, inplace=True)

    control = "control"

    adata_mix = adata.copy()
    sc.pp.normalize_total(adata_mix, target_sum=1e4)
    sc.pp.log1p(adata_mix)

    hgenes = set()
    for i in set(adata_mix.obs["gene"]) - {control}:
        try:
            adata_sub = adata_mix[adata_mix.obs["gene"].isin([i, control]), :].copy()
            sc.pp.highly_variable_genes(adata_sub, min_disp=0.2)
            hgene = set(adata_sub[:, adata_sub.var["highly_variable"]].var["ensembl_id"])
            hgenes.update(hgene)
            del adata_sub
            gc.collect()
        except Exception:
            continue
    adata_mix.var["highly_variable"] = adata_mix.var["ensembl_id"].isin(hgenes)
    adata_mix = adata_mix[:, adata_mix.var["highly_variable"]].copy()

    adata_mix.obs["perturbation"] = "Perturbed"
    adata_mix.obs.loc[adata_mix.obs["gene"] == control, "perturbation"] = "NT"
    adata_mix.obs["gene"] = list(adata_mix.obs["gene"])
    adata_mix.obs.loc[adata_mix.obs["gene"] == control, "gene"] = "NT"

    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(adata_mix, "perturbation", "NT")
    adata_mix.layers["X_pert"] = sp.sparse.csr_matrix(adata_mix.layers["X_pert"])
    mixscape_identifier.mixscape(
        adata=adata_mix, control="NT", labels="gene",
        perturbation_type="OE", layer="X_pert"
    )

    adata.obs = adata_mix.obs
    hgene = list(adata_mix.var["ensembl_id"])
    adata.var["highly_variable"] = adata.var["ensembl_id"].isin(hgene)

    adata_exnp = adata[adata.obs["mixscape_class_global"] != "NP"].copy()

    sc.pp.normalize_total(adata_exnp, target_sum=1e4)
    sc.pp.log1p(adata_exnp)

    adata_exnp.write(out_path)
    print(f"  Saved preprocessed data: {out_path} ({adata_exnp.shape})")
    del adata, adata_mix, adata_exnp
    gc.collect()


if __name__ == "__main__":
    main()
