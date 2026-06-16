import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse

CWD = str(Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(CWD, "attribution_analysis/captum_union")
os.makedirs(OUT_DIR, exist_ok=True)

STUDIES = {
    "Norman": {
        "study_full": "NormanWeissman2019_filtered_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3",
        "tfmodisco_base": "tfmodisco",
        "attr_base": "attribution",
        "perturbations": [
            "Norman.CEBPA", "Norman.CEBPB", "Norman.ETS2", "Norman.SPI1",
            "Norman.TP73", "Norman.IRF1", "Norman.JUN", "Norman.HNF4A",
            "Norman.EGR1", "Norman.FOXA1", "Norman.AHR", "Norman.SNAI1",
        ],
    },
    "Martin": {
        "study_full": "MartinRufino2025_mixscape_exnp_train__alphagenome_transfer_epoch100_batch256_adamw5e3",
        "tfmodisco_base": "tfmodisco",
        "attr_base": "attribution",
        "perturbations": [
            "MartinRufino.KLF1", "MartinRufino.NFE2", "MartinRufino.FOSL1",
            "MartinRufino.GATA1", "MartinRufino.RUNX1", "MartinRufino.TAL1",
            "MartinRufino.MYB", "MartinRufino.SPI1", "MartinRufino.GATA2",
            "MartinRufino.BCL11A",
        ],
    },
}

JASPAR_CLUSTERS = os.path.join(CWD, "reference/jaspar/clusters.tab")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qval", type=float, default=0.05,
                        help="TOMTOM q-value threshold (default: 0.05). Set <=0 to disable.")
    return parser.parse_args()


def load_jaspar_clusters(path):
    cluster = pd.read_csv(path, sep="\t", usecols=[0, 2])
    cluster = (
        cluster.set_index("cluster")["name"]
        .str.split(",", expand=True).stack()
        .str.upper().str.strip()
        .reset_index().rename(columns={0: "cluster_gene"}).drop("level_1", axis=1)
    )
    cluster = (
        cluster.set_index("cluster")["cluster_gene"]
        .str.split("::", expand=True).stack()
        .str.strip()
        .reset_index().rename(columns={0: "cluster_gene"}).drop("level_1", axis=1)
    )
    return cluster.drop_duplicates()


def get_best_match_per_pattern(ma_df):
    return ma_df.sort_values("qval").drop_duplicates(subset=["pattern"], keep="first")


def parse_motif_gene(match_str):
    if pd.isna(match_str):
        return None
    parts = match_str.split("_", 1)
    if len(parts) < 2:
        return None
    return parts[1].split(".", 2)[2] if parts[1].count(".") >= 2 else None


def check_gene_match(motif_gene, pert_gene):
    if pd.isna(motif_gene) or pd.isna(pert_gene):
        return False
    mg = set(g.strip().upper() for g in str(motif_gene).split("::"))
    pg = set(g.strip().upper() for g in str(pert_gene).split("_"))
    return bool(mg & pg)


def check_cluster_match(motif_gene, pert_gene, cluster_lookup):
    if pd.isna(motif_gene) or pd.isna(pert_gene):
        return False
    mg = set(g.strip().upper() for g in str(motif_gene).split("::"))
    pg = set(g.strip().upper() for g in str(pert_gene).split("_"))
    mc = set()
    for g in mg:
        mc.update(cluster_lookup.get(g, set()))
    pc = set()
    for g in pg:
        pc.update(cluster_lookup.get(g, set()))
    return bool(mc & pc)


def extract_seqlets_for_pert(study_full, tfmodisco_base, attr_base, pert_name, qval_threshold=None):
    tfmodisco_h5_path = os.path.join(CWD, tfmodisco_base, study_full, pert_name, f"{pert_name}_modisco_v2.h5")
    ma_list_path = os.path.join(CWD, tfmodisco_base, study_full, pert_name, "modisco_result", f"{pert_name}_MA_list.txt")
    attr_h5_path = os.path.join(CWD, attr_base, study_full, pert_name, f"{pert_name}.h5")

    for p, label in [(tfmodisco_h5_path, "modisco h5"), (attr_h5_path, "attribution h5"), (ma_list_path, "MA_list")]:
        if not os.path.exists(p):
            print(f"  [SKIP] {label} not found: {p}")
            return None, None

    with h5py.File(attr_h5_path, "r") as af:
        meta_gene = np.array([g.decode() if isinstance(g, bytes) else g for g in af["metadata/gene"][:]])
        meta_chrom = np.array([c.decode() if isinstance(c, bytes) else c for c in af["metadata/chromosome"][:]])
        meta_start = af["metadata/window_start"][:]
        meta_end = af["metadata/window_end"][:]

    ma_df = pd.read_csv(ma_list_path, sep="\t")

    if qval_threshold is not None and qval_threshold > 0:
        ma_df_filtered = ma_df[ma_df["qval"] < qval_threshold]
    else:
        ma_df_filtered = ma_df

    best_matches = get_best_match_per_pattern(ma_df_filtered)
    best_match_dict = {}
    for _, row in best_matches.iterrows():
        best_match_dict[row["pattern"]] = {
            "best_match": row["match"],
            "best_qval": row["qval"],
            "best_motif_id": row["match"].split("_")[0],
            "best_motif_gene": parse_motif_gene(row["match"]),
        }

    rows = []
    with h5py.File(tfmodisco_h5_path, "r") as mf:
        for group_name in ["pos_patterns", "neg_patterns"]:
            if group_name not in mf:
                continue
            grp = mf[group_name]
            for pat_name in sorted(grp.keys()):
                if not pat_name.startswith("pattern_"):
                    continue
                pattern_id = f"{group_name}.{pat_name}"
                pat = grp[pat_name]
                if "seqlets" not in pat:
                    continue
                seqlets = pat["seqlets"]
                example_idx = seqlets["example_idx"][:]
                start_arr = seqlets["start"][:]
                end_arr = seqlets["end"][:]
                is_revcomp = seqlets["is_revcomp"][:]
                n_seqlets = int(seqlets["n_seqlets"][0])

                bm = best_match_dict.get(pattern_id, {})

                for i in range(len(example_idx)):
                    idx = int(example_idx[i])
                    if idx >= len(meta_gene):
                        continue
                    ws = int(meta_start[idx])
                    rows.append({
                        "perturbation": pert_name,
                        "pattern": pattern_id,
                        "n_seqlets_in_pattern": n_seqlets,
                        "seqlet_idx": i,
                        "chromosome": meta_chrom[idx],
                        "genomic_start": ws + int(start_arr[i]),
                        "genomic_end": ws + int(end_arr[i]),
                        "is_revcomp": bool(is_revcomp[i]),
                        "gene": meta_gene[idx],
                        "window_start": ws,
                        "window_end": int(meta_end[idx]),
                        "best_match": bm.get("best_match"),
                        "best_qval": bm.get("best_qval"),
                        "best_motif_gene": bm.get("best_motif_gene"),
                    })

    seqlet_df = pd.DataFrame(rows) if rows else None

    ma_df_full = ma_df_filtered.copy()
    ma_df_full["motif_gene"] = ma_df_full["match"].apply(parse_motif_gene)

    return seqlet_df, ma_df_full


def main():
    args = parse_args()
    qval_threshold = args.qval

    print(f"Q-value threshold: {qval_threshold}")
    print(f"Output directory: {OUT_DIR}")

    print("Loading JASPAR clusters...")
    cluster_df = load_jaspar_clusters(JASPAR_CLUSTERS)
    cluster_lookup = {}
    for _, row in cluster_df.iterrows():
        cluster_lookup.setdefault(row["cluster_gene"], set()).add(row["cluster"])

    all_seqlets = []
    all_motif_maps = []

    for study_label, study_info in STUDIES.items():
        study_full = study_info["study_full"]
        tfmodisco_base = study_info["tfmodisco_base"]
        attr_base = study_info["attr_base"]
        perturbations = study_info["perturbations"]

        print(f"\n=== {study_label}: {len(perturbations)} perturbations ===")

        for pert in perturbations:
            print(f"  Processing {pert}...", flush=True)
            seqlet_df, motif_map = extract_seqlets_for_pert(
                study_full, tfmodisco_base, attr_base, pert,
                qval_threshold=qval_threshold,
            )
            if seqlet_df is not None:
                seqlet_df["study"] = study_label
                seqlet_df["pert_gene"] = pert.split(".")[1]
                seqlet_df["gene_match"] = seqlet_df.apply(
                    lambda r: check_gene_match(r["best_motif_gene"], r["pert_gene"]), axis=1
                )
                seqlet_df["cluster_match"] = seqlet_df.apply(
                    lambda r: check_cluster_match(r["best_motif_gene"], r["pert_gene"], cluster_lookup), axis=1
                )
                all_seqlets.append(seqlet_df)

            if motif_map is not None:
                motif_map["study"] = study_label
                motif_map["perturbation"] = pert
                pert_gene = pert.split(".")[1]
                motif_map["pert_gene"] = pert_gene
                motif_map["gene_match"] = motif_map["motif_gene"].apply(
                    lambda mg: check_gene_match(mg, pert_gene)
                )
                motif_map["cluster_match"] = motif_map["motif_gene"].apply(
                    lambda mg: check_cluster_match(mg, pert_gene, cluster_lookup)
                )
                all_motif_maps.append(motif_map)

    if not all_seqlets:
        print("No data found!")
        return

    combined_seqlets = pd.concat(all_seqlets, ignore_index=True)
    combined_motifs = pd.concat(all_motif_maps, ignore_index=True)

    out_cols = [
        "study", "perturbation", "pert_gene", "pattern", "n_seqlets_in_pattern",
        "seqlet_idx", "chromosome", "genomic_start", "genomic_end", "is_revcomp",
        "gene", "window_start", "window_end",
        "best_match", "best_qval", "best_motif_gene",
        "gene_match", "cluster_match",
    ]
    combined_seqlets[out_cols].to_csv(os.path.join(OUT_DIR, "seqlet_loci_all.tsv"), sep="\t", index=False)
    print(f"\nSaved: seqlet_loci_all.tsv ({len(combined_seqlets):,} rows)")

    matched_patterns = set()
    for _, row in combined_motifs.iterrows():
        if row["gene_match"] or row["cluster_match"]:
            matched_patterns.add((row["study"], row["perturbation"], row["pattern"]))

    mask = combined_seqlets.apply(
        lambda r: (r["study"], r["perturbation"], r["pattern"]) in matched_patterns, axis=1
    )
    matched_seqlets = combined_seqlets[mask].copy()

    matched_motif_info = combined_motifs[combined_motifs["gene_match"] | combined_motifs["cluster_match"]].copy()
    matched_motif_info = matched_motif_info.rename(columns={
        "match": "matched_motif",
        "qval": "matched_qval",
        "motif_gene": "matched_motif_gene",
    })
    matched_motif_dedup = (
        matched_motif_info
        .sort_values("matched_qval")
        .drop_duplicates(subset=["study", "perturbation", "pattern"], keep="first")
        [["study", "perturbation", "pattern", "matched_motif", "matched_qval", "matched_motif_gene", "gene_match", "cluster_match"]]
    )
    matched_seqlets = matched_seqlets.drop(columns=["gene_match", "cluster_match"]).merge(
        matched_motif_dedup, on=["study", "perturbation", "pattern"], how="left"
    )
    matched_cols = [
        "study", "perturbation", "pert_gene", "pattern", "n_seqlets_in_pattern",
        "seqlet_idx", "chromosome", "genomic_start", "genomic_end", "is_revcomp",
        "gene", "window_start", "window_end",
        "matched_motif", "matched_qval", "matched_motif_gene",
        "gene_match", "cluster_match",
    ]
    matched_seqlets[matched_cols].to_csv(os.path.join(OUT_DIR, "seqlet_loci_matched.tsv"), sep="\t", index=False)
    print(f"Saved: seqlet_loci_matched.tsv ({len(matched_seqlets):,} rows)")

    motif_cols = [
        "study", "perturbation", "pert_gene", "pattern", "num_seqlets",
        "match", "qval", "motif_gene",
        "gene_match", "cluster_match",
    ]
    combined_motifs[motif_cols].to_csv(os.path.join(OUT_DIR, "pattern_motif_mapping.tsv"), sep="\t", index=False)
    print(f"Saved: pattern_motif_mapping.tsv ({len(combined_motifs):,} rows)")

    summary_rows = []
    for (study, pert), sub in combined_seqlets.groupby(["study", "perturbation"]):
        pert_gene = sub["pert_gene"].iloc[0]
        n_patterns = sub["pattern"].nunique()
        n_seqlets = sub.drop_duplicates(subset=["pattern", "seqlet_idx"]).shape[0]
        n_genes = sub["gene"].nunique()

        motif_sub = combined_motifs[(combined_motifs["study"] == study) & (combined_motifs["perturbation"] == pert)]
        n_unique_motifs = motif_sub["match"].nunique()
        has_gene_match = motif_sub["gene_match"].any()
        has_cluster_match = motif_sub["cluster_match"].any()

        best5 = motif_sub.sort_values("qval").drop_duplicates("match").head(5)
        top5_str = "; ".join(f"{r['match']}(q={r['qval']:.4f})" for _, r in best5.iterrows())

        n_patterns_gene_match = len(set(
            motif_sub[motif_sub["gene_match"]]["pattern"]
        ))
        n_patterns_cluster_match = len(set(
            motif_sub[motif_sub["cluster_match"]]["pattern"]
        ))

        summary_rows.append({
            "study": study,
            "perturbation": pert,
            "pert_gene": pert_gene,
            "n_patterns": n_patterns,
            "n_seqlets": n_seqlets,
            "n_target_genes": n_genes,
            "n_unique_motif_matches": n_unique_motifs,
            "n_patterns_gene_match": n_patterns_gene_match,
            "n_patterns_cluster_match": n_patterns_cluster_match,
            "has_pert_gene_match": has_gene_match,
            "has_cluster_match": has_cluster_match,
            "top5_motifs": top5_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "perturbation_motif_summary.tsv"), sep="\t", index=False)
    print(f"Saved: perturbation_motif_summary.tsv ({len(summary_df)} rows)")

    print("\n" + "=" * 70)
    print(f"SUMMARY STATISTICS  (q-value threshold: {qval_threshold})")
    print("=" * 70)
    for study in summary_df["study"].unique():
        ss = summary_df[summary_df["study"] == study]
        n = len(ss)
        ng = ss["has_pert_gene_match"].sum()
        nc = ss["has_cluster_match"].sum()
        print(f"\n{study} ({n} perturbations):")
        print(f"  Gene-level match:    {ng}/{n} ({ng/n*100:.1f}%)")
        print(f"  Cluster-level match: {nc}/{n} ({nc/n*100:.1f}%)")
        for _, r in ss.iterrows():
            flag = ""
            if r["has_pert_gene_match"]:
                flag = " [GENE MATCH]"
            elif r["has_cluster_match"]:
                flag = " [CLUSTER MATCH]"
            print(f"    {r['perturbation']}: {r['n_patterns']} patterns, {r['n_seqlets']} seqlets, {r['n_target_genes']} genes{flag}")


if __name__ == "__main__":
    main()
