#!/usr/bin/env python
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import warnings
from glob import glob

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CWD = str(Path(__file__).resolve().parents[2])
os.chdir(CWD)

_CFG_PATH = os.path.join(CWD, "scripts/attribution_evaluation/52_study_config.py")
_spec = importlib.util.spec_from_file_location("study_config_52", _CFG_PATH)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
STUDY_CONFIGS = _cfg.STUDY_CONFIGS
DATA_DIR_TPL = _cfg.DATA_DIR_TPL
TABLE_DIR_TPL = _cfg.TABLE_DIR_TPL

LOG2FC_THRESHOLD = 0.5
DELTA_THRESHOLD = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True, choices=list(STUDY_CONFIGS.keys()))
    return p.parse_args()


def load_expression_data(study_name, control_col):
    df = pd.read_csv(f"data/{study_name}.tsv", sep="\t", index_col=[0])
    ctrl = df[control_col].values
    expr_cols = [c for c in df.columns if c != "training"]
    pert_cols = [c for c in expr_cols if c != control_col]

    log2fc = {pc: np.log2((df[pc].values + 1) / (ctrl + 1)) for pc in pert_cols}
    log2fc_df = pd.DataFrame(log2fc, index=df.index)
    return df, expr_cols, pert_cols, log2fc_df


def load_seqlet_metadata(study_dir):
    meta = pd.read_csv(f"{study_dir}/seqlet_metadata.tsv", sep="\t")
    attr = pd.read_csv(f"{study_dir}/seqlet_attribution.tsv", sep="\t")
    targets = pd.read_csv(f"{study_dir}/mutation_targets.tsv", sep="\t")
    return meta, attr, targets


def load_predictions(results_dir):
    h5_files = sorted(glob(f"{results_dir}/*_predictions.h5"))
    print(f"Found {len(h5_files)} prediction files")

    all_records = []
    n_out_of_bounds = 0
    for h5_path in h5_files:
        gene = os.path.basename(h5_path).replace("_predictions.h5", "")
        try:
            with h5py.File(h5_path, "r") as f:
                wt_pred = np.array(f["wt_pred"])
                mt_pred_mean = np.array(f["mt_pred_mean"])
                expr_cols = [s.decode() if isinstance(s, bytes) else s
                             for s in f["metadata"]["expr_cols"][:]]
                target_ids = [s.decode() if isinstance(s, bytes) else s
                              for s in f["metadata"]["target_ids"][:]]
                mutation_types = [s.decode() if isinstance(s, bytes) else s
                                  for s in f["metadata"]["mutation_types"][:]]
                site_ids = [s.decode() if isinstance(s, bytes) else s
                            for s in f["metadata"]["site_ids"][:]]
                mut_starts = np.array(f["metadata"]["mut_starts"])
                mut_ends = np.array(f["metadata"]["mut_ends"])
                chroms = [s.decode() if isinstance(s, bytes) else s
                          for s in f["metadata"]["chromosomes"][:]]
                ctrl_raw = f["metadata"]["control_col"][()]
                ctrl_col = ctrl_raw.decode() if isinstance(ctrl_raw, bytes) else str(ctrl_raw)
                ctrl_idx = expr_cols.index(ctrl_col)

                for ti in range(len(target_ids)):
                    if np.array_equal(mt_pred_mean[ti], wt_pred):
                        n_out_of_bounds += 1
                        continue
                    all_records.append({
                        "gene": gene,
                        "target_id": target_ids[ti],
                        "site_id": site_ids[ti],
                        "mutation_type": mutation_types[ti],
                        "chromosome": chroms[ti],
                        "mut_start": int(mut_starts[ti]),
                        "mut_end": int(mut_ends[ti]),
                        "wt_pred": wt_pred,
                        "mt_pred": mt_pred_mean[ti],
                        "ctrl_idx": ctrl_idx,
                        "expr_cols": expr_cols,
                    })
        except Exception as e:
            print(f"  Error reading {h5_path}: {e}")

    print(f"Loaded {len(all_records)} target predictions from {len(h5_files)} genes")
    if n_out_of_bounds:
        print(f"  Excluded {n_out_of_bounds} out-of-bounds targets "
              f"(mutant prediction equal to WT)")
    return all_records


def build_site_level_table(records, pert_cols, log2fc_df, meta, attr):
    print("\n[Building site-level table]")

    site_perts = meta.groupby("site_id").agg({
        "perturbation": list,
        "pert_gene": "first",
        "matched_motif": "first",
        "matched_motif_gene": "first",
        "matched_qval": "first",
        "jaspar_id": "first",
        "gene_match": lambda x: x.any(),
        "cluster_match": lambda x: x.any(),
        "pattern": "first",
    }).to_dict("index")

    attr_lookup = {}
    if len(attr) > 0:
        for _, row in attr.iterrows():
            attr_lookup[(row["site_id"], row["perturbation"])] = row

    rows = []
    for rec in records:
        gene = rec["gene"]
        target_id = rec["target_id"]
        site_id = rec["site_id"]
        mutation_type = rec["mutation_type"]
        wt_pred = rec["wt_pred"]
        mt_pred = rec["mt_pred"]
        ctrl_idx = rec["ctrl_idx"]
        expr_cols = rec["expr_cols"]

        wt_ctrl = float(wt_pred[ctrl_idx])
        mt_ctrl = float(mt_pred[ctrl_idx])
        ctrl_effect = mt_ctrl - wt_ctrl

        site_info = site_perts.get(site_id, {})
        source_perts = site_info.get("perturbation", [])
        motif_name = site_info.get("matched_motif_gene", "")
        motif_id = site_info.get("jaspar_id", "")
        is_gene_match = site_info.get("gene_match", False)
        is_cluster_match = site_info.get("cluster_match", False)

        for pc in pert_cols:
            pi = expr_cols.index(pc)
            wt_pert = float(wt_pred[pi])
            mt_pert = float(mt_pred[pi])

            fc_wt = wt_pert - wt_ctrl
            fc_mt = mt_pert - mt_ctrl
            delta_fc = fc_mt - fc_wt
            abs_delta_fc = abs(delta_fc)
            interaction = (mt_pert - wt_pert) - (mt_ctrl - wt_ctrl)

            if gene in log2fc_df.index and pc in log2fc_df.columns:
                val = log2fc_df.loc[gene, pc]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                obs_logfc = float(val)
            else:
                obs_logfc = 0.0

            if obs_logfc > LOG2FC_THRESHOLD:
                de_label = "Up"
            elif obs_logfc < -LOG2FC_THRESHOLD:
                de_label = "Down"
            else:
                de_label = "NonDE"

            attr_info = attr_lookup.get((site_id, pc), None)
            if attr_info is None:
                for sp in source_perts:
                    attr_info = attr_lookup.get((site_id, sp), None)
                    if attr_info is not None:
                        break

            attr_sum_signed = float(attr_info["attr_sum_signed"]) if attr_info is not None else np.nan
            attr_sum_abs = float(attr_info["attr_sum_abs"]) if attr_info is not None else np.nan
            attr_mean_signed = float(attr_info["attr_mean_signed"]) if attr_info is not None else np.nan
            attr_mean_abs = float(attr_info["attr_mean_abs"]) if attr_info is not None else np.nan
            attr_max_abs = float(attr_info["attr_max_abs"]) if attr_info is not None else np.nan

            attr_is_source = attr_info is not None and (site_id, pc) in attr_lookup

            attr_sign = "pos" if attr_sum_signed > 0 else ("neg" if attr_sum_signed < 0 else "zero") \
                if not np.isnan(attr_sum_signed) else "unknown"
            fc_sign_wt = "pos" if fc_wt > 0 else ("neg" if fc_wt < 0 else "zero")
            delta_sign = "pos" if delta_fc > 0 else ("neg" if delta_fc < 0 else "zero")

            sign_combo_attr_fc = f"{attr_sign}_{fc_sign_wt}"
            sign_combo_attr_delta = f"{attr_sign}_{delta_sign}"

            rows.append({
                "target_id": target_id,
                "site_id": site_id,
                "gene": gene,
                "perturbation": pc,
                "chromosome": rec["chromosome"],
                "mut_start": rec["mut_start"],
                "mut_end": rec["mut_end"],
                "mutation_type": mutation_type,
                "motif_name": motif_name,
                "motif_id": motif_id,
                "gene_match": is_gene_match,
                "cluster_match": is_cluster_match,
                "wt_ctrl": wt_ctrl,
                "wt_pert": wt_pert,
                "mt_ctrl": mt_ctrl,
                "mt_pert": mt_pert,
                "fc_wt": fc_wt,
                "fc_mt": fc_mt,
                "delta_fc": delta_fc,
                "abs_delta_fc": abs_delta_fc,
                "ctrl_effect": ctrl_effect,
                "interaction_effect": interaction,
                "attr_sum_signed": attr_sum_signed,
                "attr_sum_abs": attr_sum_abs,
                "attr_mean_signed": attr_mean_signed,
                "attr_mean_abs": attr_mean_abs,
                "attr_max_abs": attr_max_abs,
                "attr_is_source": attr_is_source,
                "attr_sign": attr_sign,
                "fc_sign_wt": fc_sign_wt,
                "delta_sign": delta_sign,
                "sign_combo_attr_fc": sign_combo_attr_fc,
                "sign_combo_attr_delta": sign_combo_attr_delta,
                "observed_logFC": obs_logfc,
                "de_label": de_label,
            })

    site_df = pd.DataFrame(rows)
    print(f"  Site-level table: {len(site_df)} rows")
    return site_df


def build_crosspert_table(site_df):
    print("\n[Building cross-perturbation table]")

    seqlet_sites = site_df[site_df["mutation_type"] == "seqlet_mut"].copy()

    rows = []
    for (site_id, gene), group in seqlet_sites.groupby(["site_id", "gene"]):
        de_mask = group["de_label"].isin(["Up", "Down"])
        nonde_mask = group["de_label"] == "NonDE"

        de_group = group[de_mask]
        nonde_group = group[nonde_mask]

        n_de = len(de_group)
        n_nonde = len(nonde_group)

        med_abs_de = de_group["abs_delta_fc"].median() if n_de > 0 else np.nan
        med_abs_nonde = nonde_group["abs_delta_fc"].median() if n_nonde > 0 else np.nan
        med_signed_de = de_group["delta_fc"].median() if n_de > 0 else np.nan
        med_signed_nonde = nonde_group["delta_fc"].median() if n_nonde > 0 else np.nan
        med_interaction_de = de_group["interaction_effect"].median() if n_de > 0 else np.nan
        med_interaction_nonde = nonde_group["interaction_effect"].median() if n_nonde > 0 else np.nan

        rows.append({
            "site_id": site_id,
            "gene": gene,
            "motif_name": group["motif_name"].iloc[0],
            "motif_id": group["motif_id"].iloc[0],
            "n_pert_total": len(group),
            "n_de_pert": n_de,
            "n_nonde_pert": n_nonde,
            "median_abs_delta_de": med_abs_de,
            "median_abs_delta_nonde": med_abs_nonde,
            "diff_abs_delta_de_minus_nonde": (med_abs_de - med_abs_nonde) if (n_de > 0 and n_nonde > 0) else np.nan,
            "median_signed_delta_de": med_signed_de,
            "median_signed_delta_nonde": med_signed_nonde,
            "diff_signed_delta_de_minus_nonde": (med_signed_de - med_signed_nonde) if (n_de > 0 and n_nonde > 0) else np.nan,
            "median_interaction_de": med_interaction_de,
            "median_interaction_nonde": med_interaction_nonde,
        })

    crosspert_df = pd.DataFrame(rows)
    print(f"  Cross-perturbation table: {len(crosspert_df)} rows")
    return crosspert_df


def build_gene_level_table(site_df):
    print("\n[Building gene-level table]")

    seqlet_sites = site_df[site_df["mutation_type"] == "seqlet_mut"].copy()

    rows = []
    for (gene, pert), group in seqlet_sites.groupby(["gene", "perturbation"]):
        n_sites = len(group)
        obs_logfc = group["observed_logFC"].iloc[0]
        de_label = group["de_label"].iloc[0]

        max_abs_delta = group["abs_delta_fc"].max()
        sum_abs_delta = group["abs_delta_fc"].sum()
        max_signed_idx = group["abs_delta_fc"].idxmax()
        max_signed_delta = group.loc[max_signed_idx, "delta_fc"]
        sum_signed_delta = group["delta_fc"].sum()

        combos = group["sign_combo_attr_fc"].value_counts()

        row_data = {
            "perturbation": pert,
            "gene": gene,
            "observed_logFC": obs_logfc,
            "de_label": de_label,
            "n_sites": n_sites,
            "max_abs_delta": max_abs_delta,
            "sum_abs_delta": sum_abs_delta,
            "max_signed_delta": max_signed_delta,
            "sum_signed_delta": sum_signed_delta,
        }

        for combo in ["pos_pos", "pos_neg", "neg_pos", "neg_neg"]:
            row_data[f"n_{combo}"] = combos.get(combo, 0)
            mask = group["sign_combo_attr_fc"] == combo
            row_data[f"sum_abs_delta_{combo}"] = group.loc[mask, "abs_delta_fc"].sum()

        rows.append(row_data)

    gene_df = pd.DataFrame(rows)
    print(f"  Gene-level table: {len(gene_df)} rows")
    return gene_df


def build_motif_level_table(site_df, pert_prefix):
    print("\n[Building motif-level table]")

    seqlet_sites = site_df[
        (site_df["mutation_type"] == "seqlet_mut")
        & (site_df["motif_name"].notna())
        & (site_df["motif_name"] != "")
    ].copy()

    pert_tf = {}
    for pc in seqlet_sites["perturbation"].unique():
        if "." in pc:
            tf = pc.split(".")[-1]
            pert_tf[pc] = tf.split("_") if "_" in tf else [tf]
        else:
            pert_tf[pc] = [pc]

    rows = []
    for (motif, pert), group in seqlet_sites.groupby(["motif_name", "perturbation"]):
        n_sites = len(group)
        med_abs = group["abs_delta_fc"].median()
        mean_abs = group["abs_delta_fc"].mean()
        med_signed = group["delta_fc"].median()
        frac_large = (group["abs_delta_fc"] > DELTA_THRESHOLD).mean()

        de_mask = group["de_label"].isin(["Up", "Down"])
        nonde_mask = group["de_label"] == "NonDE"
        med_abs_de = group.loc[de_mask, "abs_delta_fc"].median() if de_mask.sum() > 0 else np.nan
        med_abs_nonde = group.loc[nonde_mask, "abs_delta_fc"].median() if nonde_mask.sum() > 0 else np.nan

        combos = group["sign_combo_attr_fc"].value_counts()

        tfs = pert_tf.get(pert, [])
        cognate = any(motif.upper() == tf.upper() for tf in tfs)

        rows.append({
            "perturbation": pert,
            "motif_name": motif,
            "motif_id": group["motif_id"].iloc[0],
            "n_sites": n_sites,
            "median_abs_delta": med_abs,
            "mean_abs_delta": mean_abs,
            "median_signed_delta": med_signed,
            "frac_large_abs_delta": frac_large,
            "median_abs_delta_de": med_abs_de,
            "median_abs_delta_nonde": med_abs_nonde,
            "diff_abs_delta_de_minus_nonde": (med_abs_de - med_abs_nonde)
                if (not np.isnan(med_abs_de) and not np.isnan(med_abs_nonde)) else np.nan,
            "n_pos_pos": combos.get("pos_pos", 0),
            "n_pos_neg": combos.get("pos_neg", 0),
            "n_neg_pos": combos.get("neg_pos", 0),
            "n_neg_neg": combos.get("neg_neg", 0),
            "cognate_flag": cognate,
        })

    motif_df = pd.DataFrame(rows)
    print(f"  Motif-level table: {len(motif_df)} rows")
    return motif_df


def build_perturbation_level_table(site_df, motif_df):
    print("\n[Building perturbation-level table]")

    seqlet_sites = site_df[site_df["mutation_type"] == "seqlet_mut"].copy()

    rows = []
    for pert, group in seqlet_sites.groupby("perturbation"):
        n_sites = group["site_id"].nunique()
        n_genes = group["gene"].nunique()

        med_abs = group["abs_delta_fc"].median()
        mean_abs = group["abs_delta_fc"].mean()

        de_mask = group["de_label"].isin(["Up", "Down"])
        nonde_mask = group["de_label"] == "NonDE"
        med_abs_de = group.loc[de_mask, "abs_delta_fc"].median() if de_mask.sum() > 0 else np.nan
        med_abs_nonde = group.loc[nonde_mask, "abs_delta_fc"].median() if nonde_mask.sum() > 0 else np.nan

        pert_motifs = motif_df[motif_df["perturbation"] == pert].sort_values(
            "median_abs_delta", ascending=False
        )
        cognate_rows = pert_motifs[pert_motifs["cognate_flag"] == True]
        if len(cognate_rows) > 0 and len(pert_motifs) > 0:
            cognate_rank = pert_motifs.index.get_loc(cognate_rows.index[0]) + 1 \
                if cognate_rows.index[0] in pert_motifs.index else np.nan
            cognate_rank_pct = cognate_rank / len(pert_motifs) * 100 if not np.isnan(cognate_rank) else np.nan
        else:
            cognate_rank = np.nan
            cognate_rank_pct = np.nan

        combos = group["sign_combo_attr_fc"].value_counts(normalize=True)

        rows.append({
            "perturbation": pert,
            "n_sites": n_sites,
            "n_genes": n_genes,
            "median_abs_delta": med_abs,
            "mean_abs_delta": mean_abs,
            "median_abs_delta_de": med_abs_de,
            "median_abs_delta_nonde": med_abs_nonde,
            "diff_abs_delta_de_minus_nonde": (med_abs_de - med_abs_nonde)
                if (not np.isnan(med_abs_de) and not np.isnan(med_abs_nonde)) else np.nan,
            "cognate_motif_rank": cognate_rank,
            "cognate_rank_percentile": cognate_rank_pct,
            "frac_pos_pos": combos.get("pos_pos", 0),
            "frac_pos_neg": combos.get("pos_neg", 0),
            "frac_neg_pos": combos.get("neg_pos", 0),
            "frac_neg_neg": combos.get("neg_neg", 0),
        })

    pert_df = pd.DataFrame(rows)
    print(f"  Perturbation-level table: {len(pert_df)} rows")
    return pert_df


def main():
    args = parse_args()
    cfg = STUDY_CONFIGS[args.study]
    study_name = cfg["study_name"]
    control_col = cfg["control_col"]
    pert_prefix = cfg["pert_prefix"]

    study_dir = DATA_DIR_TPL.format(study=args.study)
    results_dir = f"{study_dir}/results"
    tables_dir = TABLE_DIR_TPL.format(study=args.study)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print(f"Building evaluation tables for {args.study}")
    print(f"  results_dir: {results_dir}")
    print(f"  tables_dir : {tables_dir}")
    print("=" * 60)

    print("\n[Loading data]")
    expr_df, expr_cols, pert_cols, log2fc_df = load_expression_data(study_name, control_col)
    meta, attr, targets = load_seqlet_metadata(study_dir)
    records = load_predictions(results_dir)

    if len(records) == 0:
        print("No prediction records found. Exiting.")
        return

    site_df = build_site_level_table(records, pert_cols, log2fc_df, meta, attr)
    crosspert_df = build_crosspert_table(site_df)
    gene_df = build_gene_level_table(site_df)
    motif_df = build_motif_level_table(site_df, pert_prefix)
    pert_df = build_perturbation_level_table(site_df, motif_df)

    print("\n[Saving tables]")
    tables = {
        "site_level_table": site_df,
        "site_crosspert_table": crosspert_df,
        "gene_level_table": gene_df,
        "motif_level_table": motif_df,
        "perturbation_level_table": pert_df,
    }

    for name, df in tables.items():
        path = f"{tables_dir}/{name}.tsv"
        df.to_csv(path, sep="\t", index=False)
        print(f"  {path} ({len(df)} rows, {len(df.columns)} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
