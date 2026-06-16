from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
STUDY_FULL = "JialongJiang2024_CD8T_train__alphagenome_transfer_epoch100_batch256_adamw5e3"
ADATA_PATH = ROOT / "adata" / STUDY_FULL / "adata_real_all.h5ad"
TFM_DIR = ROOT / "attribution_analysis" / "tfmodisco" / STUDY_FULL
TFM_POS_PATH = TFM_DIR / "tfmodisco_motif_matrix_pos.tsv"
TFM_NEG_PATH = TFM_DIR / "tfmodisco_motif_matrix_neg.tsv"
CONTROL_PERT = "Jialong.CONTROL_CD3"


PAPER_RCPARAMS = {
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "svg.fonttype": "none",
}


MOTIF_TF_FAMILY: dict[str, list[str]] = {
    "NR3C1": ["NR3C1"],
    "NR3C2": ["NR3C2"],
    "Ar": ["AR"],
    "ESR1": ["ESR1"],
    "ESR2": ["ESR2"],
    "PPARG": ["PPARG"],
    "PPARA::RXRA": ["PPARA", "RXRA"],
    "Nfe2l2": ["NFE2L2"],
    "MAFK": ["MAFK"],
    "Mafg": ["MAFG"],
    "MAFF": ["MAFF"],
    "Mafb": ["MAFB"],
    "MAF": ["MAF"],
    "MAF::NFE2": ["MAF", "NFE2"],
    "MAFG::NFE2L1": ["MAFG", "NFE2L1"],
    "Bach1::Mafk": ["BACH1", "MAFK"],
    "BACH1": ["BACH1"],
    "BACH2": ["BACH2"],
    "JUN": ["JUN"],
    "JUNB": ["JUNB"],
    "JUND": ["JUND"],
    "FOS": ["FOS"],
    "FOSL1": ["FOSL1"],
    "FOSL2": ["FOSL2"],
    "FOS::JUN": ["FOS", "JUN"],
    "FOS::JUND": ["FOS", "JUND"],
    "FOSB::JUN": ["FOSB", "JUN"],
    "BATF": ["BATF"],
    "BATF3": ["BATF3"],
    "BATF::JUN": ["BATF", "JUN"],
    "ELK1": ["ELK1"],
    "ELK3": ["ELK3"],
    "ELK4": ["ELK4"],
    "ETS1": ["ETS1"],
    "ETS2": ["ETS2"],
    "ETV1": ["ETV1"],
    "ETV4": ["ETV4"],
    "ETV5": ["ETV5"],
    "ETV6": ["ETV6"],
    "ELF1": ["ELF1"],
    "ELF3": ["ELF3"],
    "SRF": ["SRF"],
    "Foxo1": ["FOXO1"],
    "Foxo3": ["FOXO3"],
    "FOXO4": ["FOXO4"],
    "STAT1": ["STAT1"],
    "STAT3": ["STAT3"],
    "IRF1": ["IRF1"],
    "IRF3": ["IRF3"],
    "IRF4": ["IRF4"],
    "NFKB1": ["NFKB1"],
    "NFKB2": ["NFKB2"],
    "REL": ["REL"],
    "RELA": ["RELA"],
    "MYC": ["MYC"],
    "E2F1": ["E2F1"],
    "E2F2": ["E2F2"],
    "E2F4": ["E2F4"],
    "TFE3": ["TFE3"],
    "TFEB": ["TFEB"],
    "HIF1A": ["HIF1A"],
}


def extract_drug_name(pert_name: str) -> str:
    name = pert_name.replace("Jialong.", "")
    name = re.sub(r"_CD3(_\d+)?$", "", name)
    name = re.sub(r"_\d+nM", "", name)
    return name.strip("_")


def is_combination_pert(pert_name: str) -> bool:
    name = pert_name.replace("Jialong.", "")
    return len(re.findall(r"_\d+nM", name)) >= 2


DRUG_PANEL_NAMES_TSV = ROOT / "data" / "drug_mechanism" / "drug_panel_names.tsv"


def load_panel_drug_names() -> list[str]:
    if not DRUG_PANEL_NAMES_TSV.exists():
        raise FileNotFoundError(
            f"Drug panel names file not found: {DRUG_PANEL_NAMES_TSV}. "
            "This is a committed data file listing the experiment compound "
            "panel (parsed from the original-study AnnData perturbation labels)."
        )
    names = set()
    with DRUG_PANEL_NAMES_TSV.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = (row.get("drug_name") or "").strip()
            if name and "_" not in name:
                names.add(name)
    return sorted(names)


def _load_tfmodisco_abs_matrix() -> pd.DataFrame:
    pos = pd.read_csv(TFM_POS_PATH, sep="\t", index_col=0)
    neg = pd.read_csv(TFM_NEG_PATH, sep="\t", index_col=0)
    motifs = sorted(set(pos.index) | set(neg.index))
    perts = sorted(set(pos.columns) | set(neg.columns))
    pos = pos.reindex(index=motifs, columns=perts, fill_value=0.0)
    neg = neg.reindex(index=motifs, columns=perts, fill_value=0.0)
    return pd.DataFrame(
        np.maximum(pos.values, neg.values),
        index=motifs, columns=perts,
    )


def load_drug_mech_data() -> dict:
    adata = sc.read_h5ad(ADATA_PATH)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    if CONTROL_PERT not in expr.index:
        raise RuntimeError(f"CONTROL pert {CONTROL_PERT} missing from adata")
    control_expr = expr.loc[CONTROL_PERT]

    print("[drug_mech._common] loading TF-MoDISco absolute matrix ...")
    tfm_abs_full = _load_tfmodisco_abs_matrix()

    candidates = [p for p in tfm_abs_full.columns if "CONTROL" not in p]
    single_perts = [
        p for p in candidates
        if not is_combination_pert(p) and p in expr.index
    ]
    tfm_abs = tfm_abs_full[single_perts]

    mu = tfm_abs.mean(axis=1)
    sd = tfm_abs.std(axis=1).replace(0, np.nan)
    tfm_z = tfm_abs.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    raw_ctrl = np.expm1(control_expr.values)
    raw_pert = np.expm1(expr.loc[single_perts].values)
    fc_pseudo_df = pd.DataFrame(
        np.log2((raw_pert + 1.0) / (raw_ctrl + 1.0)),
        index=single_perts, columns=expr.columns,
    )

    drug_names = pd.Series(
        [extract_drug_name(p) for p in single_perts],
        index=single_perts, name="drug",
    )
    print(f"[drug_mech._common]   n single perts = {len(single_perts)};  "
          f"n motifs = {tfm_abs.shape[0]}")

    return {
        "tfm_abs": tfm_abs,
        "tfm_z": tfm_z,
        "fc_pseudo_df": fc_pseudo_df,
        "drug_names": drug_names,
        "single_perts": single_perts,
    }
