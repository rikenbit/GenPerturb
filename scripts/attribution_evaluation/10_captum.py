#!/usr/bin/env python3
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
import h5py

from captum.attr import InputXGradient, IntegratedGradients, Saliency, DeepLift
from genperturb.model._genperturb import GenPerturb
from genperturb.dataloaders._genome import GenomeIntervalDataset, seq_indices_to_one_hot
from genperturb.dataloaders._alphagenome_sequence import alphagenome_indices_to_one_hot



@dataclass
class ModelConfig:
    study_name: str
    study_suffix: str
    context_length: int
    pretrained: str

    @property
    def study(self) -> str:
        return f"{self.study_name}__{self.study_suffix}"

    @property
    def _is_alphagenome_fold(self) -> bool:
        return self.pretrained.startswith("alphagenome_fold_")

    @property
    def is_lora(self) -> bool:
        return "_lora_r" in self.study_suffix

    @property
    def data_tsv(self) -> str:
        if self.pretrained == "enformer":
            return f"data/{self.study_name}_enformer.tsv"
        elif self._is_alphagenome_fold:
            return f"data/{self.study_name}_{self.pretrained}.tsv"
        return f"data/{self.study_name}.tsv"

    @property
    def data_hdf5(self) -> str:
        if self.pretrained == "enformer":
            return f"data/{self.study_name}_enformer.h5"
        elif self.pretrained == "borzoi":
            return f"data/{self.study_name}_borzoi.h5"
        elif self.pretrained.startswith("alphagenome"):
            return f"data/{self.study_name}_{self.pretrained}.h5"
        return f"data/{self.study_name}.h5"

    @property
    def bed_file(self) -> str:
        if self._is_alphagenome_fold:
            return f"fasta/{self.study_name}_{self.pretrained}.bed"
        return f"fasta/{self.study_name}.bed"

    @property
    def fasta_file(self) -> str:
        return "fasta/GRCh38.p14.genome.fa"

    @property
    def prediction_file(self) -> str:
        return f"prediction/{self.study}/prediction.npy"

    @property
    def correlation_file(self) -> str:
        return f"figures/{self.study}/cor_matrix/correlation_across_perts.txt"


@dataclass
class OutputConfig:
    base_dir: str
    study: str

    def attribution_dir(self, pert: str, suffix: str = "") -> Path:
        safe_pert = pert.replace("/", "_")
        return Path(self.base_dir) / "attribution" / self.study / f"{safe_pert}{suffix}"

    def pert_dir(self, pert: str, suffix: str = "") -> Path:
        safe_pert = pert.replace("/", "_")
        return Path(self.base_dir) / "attribution_pert" / self.study / f"{safe_pert}{suffix}"

    def seq_dir(self, gene: str, suffix: str = "") -> Path:
        return Path(self.base_dir) / "attribution_seq" / self.study / f"{gene}{suffix}"



def parse_model_type(study_suffix: str) -> Tuple[int, str]:
    if "enformer" in study_suffix:
        return 196_608, "enformer"
    elif "borzoi" in study_suffix:
        return 524_288, "borzoi"
    elif match := re.search(r"alphagenome_fold_[0-3]", study_suffix):
        return 1_048_576, match.group(0)
    elif "alphagenome" in study_suffix:
        return 1_048_576, "alphagenome"
    else:
        raise ValueError(f"Unknown model type in study_suffix: {study_suffix}")


def load_model(config: ModelConfig) -> GenPerturb:
    df = pd.read_csv(config.data_tsv, sep="\t", index_col=[0])

    if config.is_lora:
        model = GenPerturb(
            df,
            bed=config.bed_file,
            fasta=config.fasta_file,
            context_length=config.context_length,
            pretrained=config.pretrained,
            training_method="lora",
            study=config.study
        )
        model.load_lora_model()
    else:
        model = GenPerturb(
            df,
            hdf5=config.data_hdf5,
            context_length=config.context_length,
            pretrained=config.pretrained,
            training_method="transfer",
            study=config.study
        )
        model.load_model()
        model.load_pretrained_model()

    model = model.cuda()

    return model


def load_dataset(config: ModelConfig) -> GenomeIntervalDataset:
    return GenomeIntervalDataset(
        bed_file=config.bed_file,
        fasta_file=config.fasta_file,
        return_seq_indices=True,
        context_length=config.context_length
    )


def load_predictions(config: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.data_tsv, sep="\t", index_col=0)
    pred = np.load(config.prediction_file)

    value_cols = df.columns[1:]

    df_pred = pd.DataFrame(pred, columns=value_cols, index=df.index)

    df_obs = df[value_cols]
    ctrl_col = df_obs.columns[0]
    df_obs_fc = (df_obs.T - df_obs[ctrl_col]).T.drop(columns=[ctrl_col])

    return df_pred, df_obs_fc


class AttributionCalculator:
    def __init__(self, model: nn.Module, context_length: int, pretrained: str):
        self.model = model
        self.context_length = context_length
        self.pretrained = pretrained

    def _get_input_sequence(self, dataset: GenomeIntervalDataset, gene_idx: int) -> torch.Tensor:
        seq_indices = dataset[gene_idx]
        if self.pretrained.startswith("alphagenome"):
            one_hot = alphagenome_indices_to_one_hot(
                seq_indices,
                sequence_key=dataset.get_interval_key(gene_idx),
            )
        else:
            one_hot = seq_indices_to_one_hot(seq_indices).float()
        return one_hot.requires_grad_(True).cuda()

    def _get_random_baseline(self, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        if shape is None:
            random_indices = torch.randint(0, 4, size=[1, self.context_length])[0]
            return seq_indices_to_one_hot(random_indices).float().requires_grad_(True).cuda()
        else:
            return torch.zeros(shape, requires_grad=True).cuda()

    def compute(
        self,
        dataset: GenomeIntervalDataset,
        gene_idx: int,
        pert_idx: int,
        method: str = "ixg",
        additional_input: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_seq = self._get_input_sequence(dataset, gene_idx)
        baseline_seq = self._get_random_baseline()

        use_tuple_input = additional_input is not None

        if use_tuple_input:
            if not additional_input.requires_grad:
                additional_input = additional_input.clone().requires_grad_(True)
            if not additional_input.is_cuda:
                additional_input = additional_input.cuda()

            inputs = (input_seq, additional_input)
            baseline_additional = self._get_random_baseline(additional_input.shape)
            baselines = (baseline_seq, baseline_additional)
        else:
            inputs = input_seq
            baselines = baseline_seq

        if method == "ixg":
            attr_method = InputXGradient(self.model)
            attributions = attr_method.attribute(inputs, target=pert_idx)

        elif method == "ig":
            attr_method = IntegratedGradients(self.model)
            attributions, _ = attr_method.attribute(
                inputs, baselines,
                target=pert_idx,
                return_convergence_delta=True,
                internal_batch_size=1000,
                n_steps=400
            )

        elif method == "sa":
            attr_method = Saliency(self.model)
            attributions = attr_method.attribute(inputs, target=pert_idx, abs=False)

        elif method == "dl":
            attr_method = DeepLift(self.model)
            attributions, _ = attr_method.attribute(
                inputs, baselines,
                target=pert_idx,
                return_convergence_delta=True
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        if use_tuple_input:
            return (attributions[0].cpu(), attributions[1].cpu())
        else:
            return attributions.cpu()



@dataclass
class AttributionResult:
    attribution: np.ndarray                      # Raw attribution (seq_len, 4)
    attribution_fc: np.ndarray                   # Fold-change attribution (seq_len, 4)
    saliency: np.ndarray                         # Saliency scores (seq_len, 4)
    chromosome: str
    seq_start: int
    seq_end: int
    gene: str
    pert: str
    additional_attribution: Optional[np.ndarray] = None       # Raw attribution for 2nd input
    additional_attribution_fc: Optional[np.ndarray] = None    # Fold-change attribution for 2nd input
    additional_saliency: Optional[np.ndarray] = None          # Saliency for 2nd input


def save_attribution_result_to_hdf5_group(
    hf: h5py.File,
    group_name: str,
    result: AttributionResult,
    attribution_method: str = "ixg"
):
    grp = hf.create_group(group_name)
    grp.create_dataset(attribution_method, data=result.attribution)
    grp.create_dataset(f"{attribution_method}_fc", data=result.attribution_fc)
    grp.create_dataset("saliency", data=result.saliency)
    grp.attrs['chromosome'] = result.chromosome
    grp.attrs['seq_start'] = result.seq_start
    grp.attrs['seq_end'] = result.seq_end

    if result.additional_attribution is not None:
        add_grp = grp.create_group('additional_input')
        add_grp.create_dataset(attribution_method, data=result.additional_attribution)
        add_grp.create_dataset(f"{attribution_method}_fc", data=result.additional_attribution_fc)
        add_grp.create_dataset("saliency", data=result.additional_saliency)



class AttributionAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        dataset: GenomeIntervalDataset,
        df_pred: pd.DataFrame,
        df_obs_fc: pd.DataFrame,
        config: ModelConfig,
        output_config: OutputConfig,
        attribution_method: str = "ixg",
        bin_size: int = 128,
        additional_input_provider: Optional[callable] = None
    ):
        self.model = model
        self.dataset = dataset
        self.df_pred = df_pred
        self.df_obs_fc = df_obs_fc
        self.config = config
        self.output_config = output_config
        self.attribution_method = attribution_method
        self.bin_size = bin_size
        self.additional_input_provider = additional_input_provider

        self.calculator = AttributionCalculator(model, config.context_length, config.pretrained)
        self._ctrl_cache = {}

    def _get_gene_idx(self, gene: str) -> int:
        return self.dataset.df.with_row_index("row_number").filter(
            pl.col("column_4") == gene
        )["row_number"][0]

    def _get_pert_idx(self, pert: str) -> int:
        return self.df_pred.columns.get_loc(pert)

    def _get_sequence_info(self, gene_idx: int) -> Tuple[str, int, int]:
        chromosome = self.dataset.df[gene_idx, 0]
        center = int(self.dataset.df[gene_idx, 1])
        half_len = self.config.context_length // 2
        seq_start = center - half_len
        seq_end = center + half_len
        return chromosome, seq_start, seq_end

    def _get_additional_input(self, gene: str, pert: str) -> Optional[torch.Tensor]:
        if self.additional_input_provider is not None:
            return self.additional_input_provider(gene, pert)
        return None

    def _get_control_attribution(self, gene: str) -> Tuple:
        if gene not in self._ctrl_cache:
            gene_idx = self._get_gene_idx(gene)
            additional_input = self._get_additional_input(gene, "ctrl")  # control perturbation

            ctrl_ixg = self.calculator.compute(
                self.dataset, gene_idx, 0,
                method=self.attribution_method,
                additional_input=additional_input
            )
            ctrl_sa = self.calculator.compute(
                self.dataset, gene_idx, 0, method="sa",
                additional_input=additional_input
            )
            self._ctrl_cache[gene] = (ctrl_ixg, ctrl_sa)
        return self._ctrl_cache[gene]

    def compute_single_attribution(
        self,
        pert: str,
        gene: str,
        use_control_subtraction: bool = True,
        debug: bool = True
    ) -> AttributionResult:
        gene_idx = self._get_gene_idx(gene)
        pert_idx = self._get_pert_idx(pert)
        chromosome, seq_start, seq_end = self._get_sequence_info(gene_idx)

        additional_input = self._get_additional_input(gene, pert)
        use_dual_input = additional_input is not None

        attr_ixg = self.calculator.compute(
            self.dataset, gene_idx, pert_idx,
            method=self.attribution_method,
            additional_input=additional_input
        )
        attr_sa = self.calculator.compute(
            self.dataset, gene_idx, pert_idx, method="sa",
            additional_input=additional_input
        )

        if use_dual_input:
            attr_ixg_seq, attr_ixg_add = attr_ixg
            attr_sa_seq, attr_sa_add = attr_sa
        else:
            attr_ixg_seq = attr_ixg
            attr_sa_seq = attr_sa
            attr_ixg_add = None
            attr_sa_add = None

        if use_control_subtraction:
            ctrl_ixg, ctrl_sa = self._get_control_attribution(gene)

            if use_dual_input:
                ctrl_ixg_seq, ctrl_ixg_add = ctrl_ixg
                ctrl_sa_seq, ctrl_sa_add = ctrl_sa

                attr_ixg_fc_seq = attr_ixg_seq - ctrl_ixg_seq
                attr_sa_fc_seq = attr_sa_seq - ctrl_sa_seq
                attr_sa_fc_seq = attr_sa_fc_seq - attr_sa_fc_seq.mean(1, keepdims=True)

                attr_ixg_fc_add = attr_ixg_add - ctrl_ixg_add
                attr_sa_fc_add = attr_sa_add - ctrl_sa_add
                if attr_sa_fc_add.dim() >= 2:
                    attr_sa_fc_add = attr_sa_fc_add - attr_sa_fc_add.mean(1, keepdims=True)
                else:
                    attr_sa_fc_add = attr_sa_fc_add - attr_sa_fc_add.mean()
            else:
                attr_ixg_fc_seq = attr_ixg_seq - ctrl_ixg
                attr_sa_fc_seq = attr_sa_seq - ctrl_sa
                attr_sa_fc_seq = attr_sa_fc_seq - attr_sa_fc_seq.mean(1, keepdims=True)
                attr_ixg_fc_add = None
                attr_sa_fc_add = None
        else:
            attr_ixg_fc_seq = attr_ixg_seq
            attr_sa_fc_seq = attr_sa_seq - attr_sa_seq.mean(1, keepdims=True)
            if use_dual_input:
                attr_ixg_fc_add = attr_ixg_add
                if attr_sa_add.dim() >= 2:
                    attr_sa_fc_add = attr_sa_add - attr_sa_add.mean(1, keepdims=True)
                else:
                    attr_sa_fc_add = attr_sa_add - attr_sa_add.mean()
            else:
                attr_ixg_fc_add = None
                attr_sa_fc_add = None

        if debug:
            attr_fc_np = attr_ixg_fc_seq.detach().numpy()
            abs_sum = np.abs(attr_fc_np).sum(axis=1)
            print(f"\n    [DEBUG] Attribution stats for {gene}:")
            print(f"    - Attribution fc abs sum range: {abs_sum.min():.6f} to {abs_sum.max():.6f}")
            print(f"    - Attribution fc abs sum mean: {abs_sum.mean():.6f}, std: {abs_sum.std():.6f}")
            if use_dual_input:
                print(f"    - Additional input shape: {attr_ixg_add.shape}")

        result = AttributionResult(
            attribution=attr_ixg_seq.detach().numpy(),
            attribution_fc=attr_ixg_fc_seq.detach().numpy(),
            saliency=attr_sa_fc_seq.detach().numpy(),
            chromosome=chromosome,
            seq_start=seq_start,
            seq_end=seq_end,
            gene=gene,
            pert=pert
        )

        if use_dual_input:
            result.additional_attribution = attr_ixg_add.detach().numpy()
            result.additional_attribution_fc = attr_ixg_fc_add.detach().numpy()
            result.additional_saliency = attr_sa_fc_add.detach().numpy()

        return result



def run_variable_genes_analysis(
    analyzer: AttributionAnalyzer,
    perturbations: List[str],
    n_top_genes: int = 150,
    suffix: str = ""
):
    print("=" * 70)
    print("Variable Genes Analysis Mode")
    print("=" * 70)

    for pert in perturbations:
        print(f"\nProcessing perturbation: {pert}")

        top_genes = list(dict.fromkeys(
            analyzer.df_obs_fc.abs().sort_values(
                pert, ascending=False
            ).loc[:, pert].head(n_top_genes * 2).index.tolist()
        ))[:n_top_genes]

        safe_pert = pert.replace("/", "_")
        output_dir = analyzer.output_config.attribution_dir(safe_pert, suffix)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = output_dir / f"{safe_pert}{suffix}_raw_attribution.h5"

        with h5py.File(h5_path, 'w') as hf:
            hf.attrs['pert'] = pert
            hf.attrs['study'] = analyzer.config.study
            hf.attrs['attribution_method'] = analyzer.attribution_method
            hf.attrs['context_length'] = analyzer.config.context_length
            hf.attrs['mode'] = 'variable_genes'

            for i, gene in enumerate(top_genes):
                print(f"\n  [{i+1}/{len(top_genes)}] Computing: {pert} x {gene}")

                analyzer._ctrl_cache.clear()

                result = analyzer.compute_single_attribution(pert, gene, debug=True)

                save_attribution_result_to_hdf5_group(
                    hf, gene, result, analyzer.attribution_method
                )

                torch.cuda.empty_cache()

        print(f"\n  Saved raw attribution HDF5: {h5_path}")
        print(f"  Output: {output_dir}")


def run_across_genes_analysis(
    analyzer: AttributionAnalyzer,
    perturbations: List[str],
    genes: List[str],
    suffix: str = "",
    debug: bool = False
):
    print("=" * 70)
    print("Across Genes Analysis Mode")
    print("=" * 70)

    single_pert_mode = len(perturbations) == 1

    for pert in perturbations:
        print(f"\nProcessing perturbation: {pert}")

        safe_pert = pert.replace("/", "_")
        output_dir = analyzer.output_config.pert_dir(pert, suffix)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = output_dir / f"{safe_pert}_raw_attribution.h5"

        with h5py.File(h5_path, 'w') as hf:
            hf.attrs['pert'] = pert
            hf.attrs['study'] = analyzer.config.study
            hf.attrs['attribution_method'] = analyzer.attribution_method
            hf.attrs['context_length'] = analyzer.config.context_length
            hf.attrs['mode'] = 'across_genes'

            import gc as _gc

            for i, gene in enumerate(genes):
                if debug:
                    print(f"\n  [{i+1}/{len(genes)}] Computing: {pert} x {gene}")
                else:
                    print(f"  [{i+1}/{len(genes)}] Computing: {pert} x {gene}", end="")

                try:
                    result = analyzer.compute_single_attribution(pert, gene, debug=debug)

                    save_attribution_result_to_hdf5_group(
                        hf, gene, result, analyzer.attribution_method
                    )

                    if not debug:
                        print(f" ... done")

                    torch.cuda.empty_cache()

                    del result
                    if single_pert_mode:
                        analyzer._ctrl_cache.pop(gene, None)
                    if (i + 1) % 100 == 0:
                        hf.flush()
                        _gc.collect()

                except Exception as e:
                    print(f" ... ERROR: {e}")
                    continue

            hf.flush()

        print(f"\n  Saved raw attribution HDF5: {h5_path}")
        print(f"  Output: {output_dir}")


def run_across_perturbations_analysis(
    analyzer: AttributionAnalyzer,
    genes: List[str],
    perturbations: List[str],
    suffix: str = "",
    debug: bool = False
):
    print("=" * 70)
    print("Across Perturbations Analysis Mode")
    print("=" * 70)

    for gene in genes:
        print(f"\nProcessing gene: {gene}")

        output_dir = analyzer.output_config.seq_dir(gene, suffix)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = output_dir / f"{gene}_raw_attribution.h5"

        gene_idx = analyzer._get_gene_idx(gene)
        chromosome, seq_start, seq_end = analyzer._get_sequence_info(gene_idx)

        with h5py.File(h5_path, 'w') as hf:
            hf.attrs['gene'] = gene
            hf.attrs['chromosome'] = chromosome
            hf.attrs['seq_start'] = seq_start
            hf.attrs['seq_end'] = seq_end
            hf.attrs['study'] = analyzer.config.study
            hf.attrs['attribution_method'] = analyzer.attribution_method
            hf.attrs['context_length'] = analyzer.config.context_length
            hf.attrs['mode'] = 'across_perturbations'

            for i, pert in enumerate(perturbations):
                if debug:
                    print(f"\n  [{i+1}/{len(perturbations)}] Computing: {pert} x {gene}")
                else:
                    print(f"  [{i+1}/{len(perturbations)}] Computing: {pert} x {gene}", end="")

                try:
                    result = analyzer.compute_single_attribution(pert, gene, debug=debug)

                    safe_pert = pert.replace("/", "_")
                    save_attribution_result_to_hdf5_group(
                        hf, safe_pert, result, analyzer.attribution_method
                    )
                    hf[safe_pert].attrs['pert'] = pert

                    if not debug:
                        print(f" ... done")

                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f" ... ERROR: {e}")
                    continue

        print(f"\n  Saved raw attribution HDF5: {h5_path}")
        print(f"  Output: {output_dir}")



def get_tf_list() -> List[str]:
    tf_file = "reference/humantfs/DatabaseExtract_v_1.01.txt"
    return pd.read_csv(tf_file, sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].tolist()


_CONDITION_PERTS = {
    "NormanWeissman2019": [
        "Norman.IRF1", "Norman.TP73", "Norman.CEBPA", "Norman.HNF4A",
        "Norman.FOXA1", "Norman.AHR", "Norman.PRDM1", "Norman.SPI1",
        "Norman.SNAI1", "Norman.KMT2A", "Norman.CEBPB", "Norman.JUN",
        "Norman.ETS2", "Norman.EGR1",
    ],
    "MartinRufino2025_mixscape_exnp": [
        "MartinRufino.BCL11A", "MartinRufino.FOSL1", "MartinRufino.GATA1",
        "MartinRufino.GATA2", "MartinRufino.GFI1B", "MartinRufino.KLF1",
        "MartinRufino.LDB1", "MartinRufino.LMO2", "MartinRufino.MYB",
        "MartinRufino.NFE2", "MartinRufino.RUNX1", "MartinRufino.SPI1",
        "MartinRufino.TAL1",
    ],
}


def select_perturbations(pert_frame: pd.DataFrame, target: str, study_name: str = "") -> List[str]:
    if target == "all":
        return pert_frame.columns.tolist()
    elif target == "tf":
        tf_list = get_tf_list()
        pattern = "|".join(tf_list)
        return pert_frame.T[pert_frame.columns.str.contains(pattern)].index.tolist()
    elif target == "condition":
        for dataset_key, perts in _CONDITION_PERTS.items():
            if dataset_key in study_name:
                return perts
        return pert_frame.columns.tolist()
    else:
        return pert_frame.columns.tolist()


def select_genes(config: ModelConfig, target: str, training_split: str = "test") -> List[str]:
    cor = pd.read_csv(config.correlation_file, sep="\t")

    if target == "all":
        return cor.query(f'training == "{training_split}"')["Gene"].tolist()
    elif target == "top":
        return cor.query(f'training == "{training_split}"').head(10)["Gene"].tolist()
    elif target == "condition":
        return ['ECH1', 'HNRNPL', 'RINL', 'NFKBIB', 'SIRT2']  # Example genes
    else:
        return cor.query(f'training == "{training_split}"')["Gene"].tolist()



def example_additional_input_provider(gene: str, pert: str) -> torch.Tensor:
    feature_dim = 128  # Example dimension


    return torch.randn(1, feature_dim, requires_grad=True)



def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    study_name = sys.argv[1]
    study_suffix = sys.argv[2]
    mode = sys.argv[3]  # variable_genes, across_genes, across_perturbations
    target = sys.argv[4]  # all, tf, condition, test, etc.

    single_pert = None
    for i, arg in enumerate(sys.argv):
        if arg == "--pert" and i + 1 < len(sys.argv):
            single_pert = sys.argv[i + 1]
            break

    base_dir = "."
    for i, arg in enumerate(sys.argv):
        if arg == "--base-dir" and i + 1 < len(sys.argv):
            base_dir = sys.argv[i + 1]
            break

    gene_list_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--gene-list" and i + 1 < len(sys.argv):
            gene_list_file = sys.argv[i + 1]
            break

    output_suffix = ""
    for i, arg in enumerate(sys.argv):
        if arg == "--output-suffix" and i + 1 < len(sys.argv):
            output_suffix = sys.argv[i + 1]
            break

    use_dual_input = "--dual-input" in sys.argv

    torch.manual_seed(123)
    np.random.seed(123)

    context_length, pretrained = parse_model_type(study_suffix)

    config = ModelConfig(
        study_name=study_name,
        study_suffix=study_suffix,
        context_length=context_length,
        pretrained=pretrained
    )

    output_config = OutputConfig(
        base_dir=base_dir,
        study=config.study
    )

    print("=" * 70)
    print(f"Attribution Analysis")
    print("=" * 70)
    print(f"Study: {config.study}")
    print(f"Mode: {mode}")
    print(f"Target: {target}")
    print(f"Context length: {context_length}")
    print(f"Pretrained model: {pretrained}")
    print(f"Dual input mode: {use_dual_input}")
    if single_pert:
        print(f"Single perturbation: {single_pert}")
    print("=" * 70)

    print("\nLoading model...")
    model = load_model(config)

    print("Loading dataset...")
    dataset = load_dataset(config)

    print("Loading predictions...")
    df_pred, df_obs_fc = load_predictions(config)

    additional_input_provider = None
    if use_dual_input:
        print("Dual input mode enabled - using example_additional_input_provider")
        additional_input_provider = example_additional_input_provider

    analyzer = AttributionAnalyzer(
        model=model,
        dataset=dataset,
        df_pred=df_pred,
        df_obs_fc=df_obs_fc,
        config=config,
        output_config=output_config,
        attribution_method="ixg",
        bin_size=128,
        additional_input_provider=additional_input_provider
    )

    if mode == "variable_genes":
        perturbations = select_perturbations(df_obs_fc, target, study_name=config.study)
        if single_pert:
            assert single_pert in perturbations, f"--pert {single_pert} not in selected perturbations: {perturbations}"
            perturbations = [single_pert]
        print(f"\nSelected {len(perturbations)} perturbations for variable genes analysis")
        run_variable_genes_analysis(analyzer, perturbations, n_top_genes=200)

    elif mode == "union_genes":
        assert gene_list_file is not None, "--gene-list is required for union_genes mode"
        assert single_pert is not None, "--pert is required for union_genes mode"

        with open(gene_list_file) as f:
            genes = [line.strip() for line in f if line.strip()]

        perturbations = select_perturbations(df_obs_fc, target, study_name=config.study)
        assert single_pert in perturbations, f"--pert {single_pert} not in selected perturbations: {perturbations}"
        perturbations = [single_pert]

        suffix = output_suffix if output_suffix else "_union"
        print(f"\nUnion genes mode: {len(genes)} genes, pert={single_pert}, suffix={suffix}")

        safe_pert = single_pert.replace("/", "_")
        output_dir = analyzer.output_config.attribution_dir(safe_pert, suffix="")
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = output_dir / f"{safe_pert}{suffix}_raw_attribution.h5"

        print(f"  Output: {h5_path}")

        with h5py.File(h5_path, 'w') as hf:
            hf.attrs['pert'] = single_pert
            hf.attrs['study'] = analyzer.config.study
            hf.attrs['attribution_method'] = analyzer.attribution_method
            hf.attrs['context_length'] = analyzer.config.context_length
            hf.attrs['mode'] = 'union_genes'

            import gc as _gc
            for i, gene in enumerate(genes):
                print(f"  [{i+1}/{len(genes)}] Computing: {single_pert} x {gene}", end="")
                try:
                    result = analyzer.compute_single_attribution(single_pert, gene, debug=False)
                    save_attribution_result_to_hdf5_group(
                        hf, gene, result, analyzer.attribution_method
                    )
                    print(f" ... done")
                    torch.cuda.empty_cache()
                    del result
                    if (i + 1) % 100 == 0:
                        hf.flush()
                        _gc.collect()
                except Exception as e:
                    print(f" ... ERROR: {e}")
                    continue
            hf.flush()

        print(f"\n  Saved: {h5_path}")

    elif mode == "across_genes":
        cor = pd.read_csv(config.correlation_file, sep="\t")

        if target == "tf_allgene":
            suffix = ".all"
            perturbations = select_perturbations(df_obs_fc, "tf")
            genes = cor["Gene"].tolist()
        elif target == "allgene":
            suffix = ".all"
            perturbations = df_obs_fc.columns.tolist()
            genes = cor["Gene"].tolist()
        elif target == "test":
            suffix = ".test"
            perturbations = [
                "Norman.HNF4A", "Norman.IRF1", "Norman.CEBPA", "Norman.TP73",
                "Norman.FOXA1", "Norman.AHR", "Norman.PRDM1", "Norman.SPI1",
                "Norman.SNAI1", "Norman.KMT2A", "Norman.CEBPB", "Norman.JUN",
                "Norman.ETS2", "Norman.EGR1"
            ]
            genes = cor.query('training == "test"')["Gene"].tolist()
        else:
            suffix = ""
            perturbations = select_perturbations(df_obs_fc, target, study_name=config.study)
            genes = cor["Gene"].tolist()

        if single_pert:
            assert single_pert in perturbations, f"--pert {single_pert} not in selected perturbations: {perturbations}"
            perturbations = [single_pert]

        print(f"\nSelected {len(perturbations)} perturbations, {len(genes)} genes")
        run_across_genes_analysis(analyzer, perturbations, genes, suffix)

    elif mode == "across_perturbations":
        perturbations = df_obs_fc.columns.tolist()
        genes = select_genes(config, target)

        print(f"\nSelected {len(genes)} genes, {len(perturbations)} perturbations")
        run_across_perturbations_analysis(analyzer, genes, perturbations)

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: variable_genes, across_genes, across_perturbations")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
