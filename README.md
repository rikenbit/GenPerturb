# GenPerturb

GenPerturb is a model-based interpretation framework for Perturb-seq data. It
fits pretrained sequence representations to observed perturbation responses and
uses control-subtracted input-gradient attribution to prioritize sequence-level
cis-regulatory hypotheses, including candidate enhancers, motifs, and regulatory
programs.

Run commands from the repository root. Start with
[`reproducibility/RUN_ORDER.md`](reproducibility/RUN_ORDER.md) to regenerate
manuscript analyses from public datasets and reference resources.
Required files and fixed parameters are listed in
[`reproducibility/INPUTS_AND_OUTPUTS.md`](reproducibility/INPUTS_AND_OUTPUTS.md).

## Repository layout

| Directory/file | Contents |
|---|---|
| `genperturb/` | Preprocessing, model, training and evaluation code |
| `scripts/` | Analysis and source-panel plotting scripts |
| `pipeline/` | Preprocessing, training and analysis launchers |
| `reproducibility/` | Commands, input filenames and fixed analysis parameters |
| `environments/` | Installable Conda package specifications and post-install script |
| `dataset_model_config.py` | Study/model identifiers and display settings |

## Environment setup

Create the environments required for your analysis:

```bash
conda env create -f environments/singlecell.yml
conda env create -f environments/alphagenome.yml
conda env create -f environments/modisco.yml
conda env create -f environments/gimme.yml
conda run -n modisco bash environments/post_install/patch_modisco_report_nan_guard.sh
```

Borzoi, Enformer and SimpleCNN specifications are also provided under
`environments/`. See [`environments/README.md`](environments/README.md) for the
step-to-environment mapping.

## Input preparation

Prepare the public datasets, GRCh38 reference FASTA, GENCODE annotations,
motif databases and pretrained model weights. Data acquisition commands are
listed in `genperturb/preprocess/01_download_data.sh`. Required paths and
pipeline-generated files are listed in
[`INPUTS_AND_OUTPUTS.md`](reproducibility/INPUTS_AND_OUTPUTS.md).

```bash
mkdir -p data/adata data/MartinRufino fasta reference
```

Register the reference genome for GimmeMotifs after preparing the FASTA:

```bash
conda activate gimme
genomepy install -p local fasta/GRCh38.p14.genome.fa
```

## Cluster configuration

Set the Conda initialization path and SLURM resources for your system:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export PARTITION_GPU=gpu
export PARTITION_CPU=cpu
export GPU_GRES=gpu:1
```

`pipeline/_common.sh` defines memory defaults, study lists and model suffixes.
Use `STUDIES_OVERRIDE` to select a study, for example:

```bash
STUDIES_OVERRIDE="NormanWeissman2019_filtered_mixscape_exnp_train" \
  bash pipeline/01_training/01a_transfer.sh alphagenome
```

## Run the analyses

Follow [`RUN_ORDER.md`](reproducibility/RUN_ORDER.md) in order:

1. Preprocess single-cell data, extract sequence embeddings and assemble training inputs.
2. Fit transfer models and baselines, then evaluate expression outputs.
3. Compute attribution, score enhancers and identify motifs.
4. Run in-silico mutation, lineage and compound analyses.

Wait for each SLURM stage to finish before running its downstream commands.
Generated file paths and fixed parameters are listed in
[`INPUTS_AND_OUTPUTS.md`](reproducibility/INPUTS_AND_OUTPUTS.md).
These intermediate files connect successive stages: generate them with the
listed commands, or reuse matching outputs at the specified paths. The
runbook includes gene-list preparation and complete SLURM commands for the
union-attribution stage used by the locus analyses.

## Associated paper

[GenPerturb manuscript](https://doi.org/10.64898/2026.07.01.735806)
