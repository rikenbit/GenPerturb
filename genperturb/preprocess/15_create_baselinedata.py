#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error


def _locate_reference_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    try:
        idx = cols.index("training") + 1
    except ValueError:
        raise ValueError("'training' column not found.")
    if idx >= len(cols):
        raise ValueError("No column exists to the right of 'training'.")
    return cols[idx]


def save_prediction(df_pred: pd.DataFrame, study_tag: str) -> None:
    out_dir = Path("prediction") / study_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "prediction.npy",
            df_pred.iloc[:, 1:].to_numpy(dtype=np.float32))


def make_baseline_control(df: pd.DataFrame, noise_level: float = 0.001) -> pd.DataFrame:
    ref_col = _locate_reference_column(df)
    cols_to_fill = [c for c in df.columns if c != "training"]

    mat = np.repeat(
        df[ref_col].to_numpy()[:, None],
        len(cols_to_fill), axis=1
    ).astype(np.float32)

    noise = np.random.normal(
        loc=0.0, scale=noise_level, size=mat.shape
    ).astype(np.float32)
    mat += noise

    out = df.copy()
    out[cols_to_fill] = mat
    return out


def make_baseline_peturbmean(df: pd.DataFrame, noise_level: float = 0.001) -> pd.DataFrame:
    ref_col = _locate_reference_column(df)
    cols_to_fill = [c for c in df.columns if c != "training"]
    cols_to_avg  = [c for c in cols_to_fill if c != ref_col]

    row_means = df[cols_to_avg].mean(axis=1).to_numpy(dtype=np.float32)
    mat = np.repeat(row_means[:, None], len(cols_to_fill), axis=1)

    noise = np.random.normal(
        loc=0.0, scale=noise_level, size=mat.shape
    ).astype(np.float32)
    mat += noise

    # Keep the original values for the control column
    ref_idx = cols_to_fill.index(ref_col)
    mat[:, ref_idx] = df[ref_col].to_numpy(dtype=np.float32)

    out = df.copy()
    out[cols_to_fill] = mat
    return out


def make_baseline_shuffle(df: pd.DataFrame, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols_to_shuffle = [c for c in df.columns if c != "training"]

    out = df.copy()
    mat = out[cols_to_shuffle].to_numpy()

    for i in range(mat.shape[0]):
        rng.shuffle(mat[i])

    out[cols_to_shuffle] = mat
    return out


class ModelStats:
    def __init__(self, study: str, df_gold: pd.DataFrame, pred_numeric: pd.DataFrame):
        self.study = study
        self.df1 = df_gold
        self.df2 = pred_numeric.copy()
        self.df2.columns = self.df1.columns[1:]
        self.df2.index   = self.df1.index

    def _save_fc_number(self, thresh: float = 0.25):
        self.fc_num = (self.df1.iloc[:, 1:].abs() >= thresh).sum(axis=1)

    def _corr_rows(self, A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
        r_vals, p_vals, slopes, mses = [], [], [], []
        for i in range(len(A)):
            try:
                _, _, r, p, _ = stats.linregress(
                    A.iloc[i, 1:].astype("float32"), B.iloc[i, :]
                )
                mse = mean_squared_error(
                    A.iloc[i, 1:].astype("float32"), B.iloc[i, :]
                )
            except Exception:
                r, p, mse = 0.0, 1.0, 0.0
                slopes.append(0.0)
            slopes.append(0.0 if np.isnan(r) else r)
            r_vals.append(r)
            p_vals.append(p)
            mses.append(mse)

        return pd.DataFrame({
            "Gene": A.index,
            "training": A["training"].values,
            "Correlation": r_vals,
            "pval": p_vals,
            "slope": slopes,
            "MSE": mses,
            "FC_gene_num": self.fc_num.values,
        })

    def _corr_cols(self, A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
        summaries = []
        for split in ["train", "val", "test"]:
            a_sub = A[A["training"] == split].iloc[:, 1:].T
            b_sub = B[B["training"] == split].iloc[:, 1:].T

            r_vals, p_vals, slopes, mses = [], [], [], []
            for i in range(len(a_sub)):
                try:
                    _, _, r, p, _ = stats.linregress(
                        a_sub.iloc[i, :], b_sub.iloc[i, :]
                    )
                    mse = mean_squared_error(a_sub.iloc[i, :], b_sub.iloc[i, :])
                except Exception:
                    r, p, mse = 0.0, 1.0, 0.0
                slopes.append(0.0 if np.isnan(r) else r)
                r_vals.append(r)
                p_vals.append(p)
                mses.append(mse)

            summaries.append(
                pd.DataFrame({
                    "Gene": a_sub.index,
                    "Correlation": r_vals,
                    "pval": p_vals,
                    "slope": slopes,
                    "MSE": mses,
                    "training": [split] * len(a_sub),
                })
            )
        return pd.concat(summaries, ignore_index=True)

    def run(self) -> None:
        out_dir = Path("figures") / self.study / "cor_matrix"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_fc_number()

        corr_perts = self._corr_rows(self.df1, self.df2)
        var_ser  = self.df1.iloc[:, 1:].var(axis=1).rename("var")
        mean_ser = self.df1.iloc[:, 1:].mean(axis=1).rename("mean")
        corr_perts = corr_perts.merge(var_ser, left_on="Gene", right_index=True)
        corr_perts = corr_perts.merge(mean_ser, left_on="Gene", right_index=True)
        corr_perts["Variance"] = pd.qcut(
            corr_perts["var"], 5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )
        corr_perts["Mean"] = pd.qcut(
            corr_perts["mean"], 5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )
        corr_perts.sort_values("Correlation", ascending=False).to_csv(
            out_dir / "correlation_across_perts.txt", sep="	", index=False
        )

        gold_plus_pred = pd.concat([self.df1[["training"]], self.df2], axis=1)
        corr_genes = self._corr_cols(self.df1, gold_plus_pred)
        corr_genes.sort_values("Correlation", ascending=False).to_csv(
            out_dir / "correlation_across_genes.txt", sep="	", index=False
        )


if __name__ == "__main__":
    studies = [
        "NormanWeissman2019_filtered_mixscape_exnp_train",
        "ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
        "ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
        "ReplogleWeissman2022_rpe1_mixscape_exnp_train",
        "JialongJiang2024_Myeloid_train",
        "JialongJiang2024_CD4T_train",
        "JialongJiang2024_CD8T_train",
        "JialongJiang2024_B_cell_train",
        "Srivatsan2019_A549_train",
        "Srivatsan2019_K562_train",
        "Srivatsan2019_MCF7_train",
        "MartinRufino2025_mixscape_exnp_train",
        "MartinRufino2025_ProErythroblast_train",
        "MartinRufino2025_PolychromaticErythroblast_train",
        "MartinRufino2025_BasophilicErythroblast_train",
        "MartinRufino2025_OrthochromaticErythroblast_train",
        "MartinRufino2025_EoBasoMastPrecursor_train",
        "MartinRufino2025_CFUE_train",
    ]

    for study in studies:
        df_path = Path("data") / f"{study}.tsv"
        if not df_path.exists():
            print(f"[warn] {df_path} not found – skipping.")
            continue

        df = pd.read_csv(df_path, sep="	", index_col=0)

        baselines = {
            "baseline_control":     make_baseline_control(df),
            "baseline_peturbmean":  make_baseline_peturbmean(df),
            "baseline_shuffle":     make_baseline_shuffle(df),
        }

        for tag, pred_df in baselines.items():
            study_tag = f"{study}__{tag}_transfer_epoch100_batch256_adamw5e3"
            print(f"* processing {study_tag}")

            save_prediction(pred_df, study_tag)

            ModelStats(study_tag, df, pred_df.iloc[:, 1:]).run()
