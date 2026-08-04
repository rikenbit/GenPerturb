# GenPerturb

GenPerturb is a model-based interpretation framework for Perturb-seq data. It
fits pretrained sequence representations to observed perturbation responses and
uses control-subtracted input-gradient attribution to prioritize sequence-level
cis-regulatory hypotheses, including candidate enhancers, motifs, and regulatory
programs.

This repository contains the code used for the manuscript analyses and the
pipeline entrypoints needed to run the framework. Generated results, model
checkpoints, large input matrices, and figure outputs are not intended to be
stored in this repository. The lower-level implementation lives under package
and analysis directories; routine use should go through the wrappers in
`pipeline/`.

## Repository Layout

```text
genperturb/, scripts/        Core preprocessing, model, training, and evaluation code
pipeline/                    Reproducible entrypoints for preprocessing, training, attribution, and manuscript analyses
environments/                Publication Conda environment specifications and post-install notes
create_env.sh                Backward-compatible pointer to environments/
```

The main pipeline phases are:

```text
pipeline/00_preprocess/      Sequence embeddings, single-cell preprocessing, training tables
pipeline/01_training/        Transfer learning and training baselines
pipeline/fig2_expression/    Expression-level manuscript summaries
pipeline/fig3_attribution/   Captum, enhancer/motif evaluation, TF-MoDISco, mutation analyses for Fig. 3
pipeline/fig4_lineage/       Resume-safe attribution and motif analysis route for lineage analyses
pipeline/fig5_drug/          Drug mechanism manuscript analysis
```

Cluster-specific defaults such as `CONDA_SH`, SLURM partitions, GPU resources,
memory, model list, and study list are centralized in `pipeline/_common.sh`.
Before running the wrappers, edit these defaults or override them from the
shell so that paths, partitions, GPU GRES strings, memory limits, and study
lists match the target cluster. The committed values intentionally use generic
placeholders for publication; they must be replaced with paths and SLURM
settings from the target system before submitting jobs.

```bash
CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh \
PARTITION_GPU=gpu \
PARTITION_CPU=cpu \
GPU_GRES=gpu:1 \
bash pipeline/01_training/01a_transfer.sh alphagenome
```

Some long-running GPU wrappers also set node constraints outside
`pipeline/_common.sh`. `NODELIST` is optional in
`pipeline/01_training/01d_lora.sh` and
`pipeline/01_training/01e_full_finetuning.sh`; set it only when the target
cluster requires a specific node. The Fig. 3 mutation wrapper uses
`MUTATION_NODES` for optional comma-separated node names.

## 1. Environment Setup

Environment setup is recorded as one curated Conda YAML file per analysis
environment under `environments/`. Create only the environments needed for the
analyses you plan to run:

```bash
conda env create -f environments/alphagenome.yml
conda env create -f environments/modisco.yml
```

The main environments are:

| Environment | Main use |
|---|---|
| `singlecell` | single-cell preprocessing, pseudobulk construction, expression-level evaluation, light aggregation |
| `alphagenome` | AlphaGenome embeddings, transfer learning, Captum attribution |
| `borzoi` | Borzoi embeddings, transfer learning, Captum attribution |
| `enformer` | Enformer embeddings, transfer learning, Captum attribution |
| `simplecnn` | From-scratch SimpleCNN baseline; depends on the Python package `bend` |
| `modisco` | TF-MoDISco motif discovery and motif report generation |
| `gimme` | GimmeMotifs motif discovery and motif enrichment summaries |


## 2. Input Data and Working Directories

Large downloaded files and derived preprocessing outputs are kept outside the
tracked source layout. Use `genperturb/preprocess/01_download_data.sh` as the
download recipe for the public Perturb-seq, multiome, reference genome,
enhancer-score, and motif-reference inputs. Review the script and run the
needed blocks from the repository root after creating the expected top-level
working directories:

```bash
mkdir -p data/adata data/MartinRufino fasta reference
less genperturb/preprocess/01_download_data.sh
```

The expected working state is:

```text
data/adata/                         Raw and processed AnnData inputs (*.h5ad)
data/MartinRufino/                  MartinRufino multiome raw h5 matrices and metadata
data/AllPredictions*.txt.gz         ABC-score input and hg38-lifted copies
data/ENCFF*.bed.gz                  rE2G / ENCODE enhancer-gene reference files
data/*.tsv                          Pseudobulk CPM and matched training expression tables
data/*_train_{model}.h5             Model-specific sequence embeddings matched to training genes
data/{model}_embedding*.npy         Genome-wide TSS embeddings extracted from pretrained models
fasta/GRCh38.p14.genome.fa          Reference genome used by embedding and attribution code
fasta/gencode.*.gff3, fasta/gencode.*.gtf
fasta/gencode.*.all_tss.bed         Representative TSS table generated from GENCODE
fasta/gencode.*.tss.bed             TSS table annotated with AlphaGenome fold labels
fasta/*_train*.bed                  Study-specific gene/TSS tables with train/val/test labels
fasta/alphagenome/all_regions.bed   AlphaGenome fold intervals used to assign split labels
reference/                          Motif databases and other external references
```

GimmeMotifs reference-genome registration is a data-preparation step, not an
environment dependency. After creating the `gimme` environment and preparing
the reference FASTA, register the local genome outside version control:

```bash
conda activate gimme
genomepy install -p local fasta/GRCh38.p14.genome.fa
```

Before embedding extraction,
`fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed` must exist.
It is derived from the downloaded GENCODE GFF3 and AlphaGenome fold intervals
by `genperturb/preprocess/03_gfftotssbed.py`,
`genperturb/preprocess/04_generate_alphagenome_folds.py`, and
`genperturb/preprocess/05_make_training_bed.py`.

In the transfer-learning path, `data/{study}_cpm.tsv` is first created from
processed single-cell data. Training assembly then joins that expression table
to `fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed` and the
pretrained embedding arrays, producing matched pairs such as
`fasta/{study}_train.bed`, `data/{study}_train.tsv`, and
`data/{study}_train_alphagenome.h5`. Borzoi and Enformer use the same
study-level expression and BED files but write model-specific HDF5 files such
as `data/{study}_train_borzoi.h5` and `data/{study}_train_enformer.h5`.

## 3. Preprocessing

Preprocessing wrappers are in `pipeline/00_preprocess/`. They prepare pretrained
sequence-model embeddings, Perturb-seq pseudobulk data, and the matched training
tables used by transfer learning.

Typical order:

```bash
# Extract sequence representations. With no argument, this runs AlphaGenome.
bash pipeline/00_preprocess/00a_embedding_extract.sh alphagenome
bash pipeline/00_preprocess/00a_embedding_extract.sh borzoi enformer

# Preprocess single-cell perturbation data and construct pseudobulk matrices.
bash pipeline/00_preprocess/00c_singlecell_preprocess.sh

# Assemble training tables from embeddings and expression data.
bash pipeline/00_preprocess/00b_build_training_data.sh

# Build baseline data tables used by expression-level comparisons.
bash pipeline/00_preprocess/00d_baseline_data.sh
```

The wrappers call the implementation code internally. 

## 4. Transfer Learning

The main transfer-learning entrypoint is
`pipeline/01_training/01a_transfer.sh`. It submits one SLURM job per study and
model. With no model argument it uses the default pretrained models defined in
`pipeline/_common.sh`: `alphagenome`, `borzoi`, and `enformer`.

```bash
# Run AlphaGenome transfer learning for all configured studies.
bash pipeline/01_training/01a_transfer.sh alphagenome

# Run all default pretrained models.
bash pipeline/01_training/01a_transfer.sh

# Rerun one study only.
STUDIES_OVERRIDE="NormanWeissman2019_filtered_mixscape_exnp_train" \
bash pipeline/01_training/01a_transfer.sh alphagenome
```

The default transfer-learning output suffix is:

```text
{model}_transfer_epoch100_batch256_adamw5e3
```

Additional training wrappers are mainly for manuscript comparisons:

| Wrapper | Purpose |
|---|---|
| `pipeline/01_training/01b_simplecnn_baseline.sh` | from-scratch SimpleCNN baseline |
| `pipeline/01_training/01c_alphagenome_folds.sh` | AlphaGenome fold-specific transfer runs for multifold comparisons |
| `pipeline/01_training/01d_lora.sh` | AlphaGenome LoRA sweep |
| `pipeline/01_training/01e_full_finetuning.sh` | AlphaGenome full-backbone finetuning |

Training outputs are written outside the tracked source tree structure as model
predictions, checkpoint logs, and evaluation summaries under the configured
project output directories.

## 5. Captum Attribution and TF-MoDISco

Attribution jobs compute input-gradient based scores from fitted models. Motif
discovery is then run from the attribution outputs with TF-MoDISco and, where
needed, GimmeMotifs.

There are two attribution output routes:

| Route | Used for |
|---|---|
| `attribution/` | Fig. 3 paper route, including Martin enhancer AUPRC and Fig. 3 motif/mutation analyses |
| `attribution_analysis/` | Fig. 4 and Fig. 5 analyses, downstream motif matrices |

Do not substitute one route for the other without checking the relevant
dependency notes.

### Fig. 3 Paper Route

For MartinRufino enhancer and motif analyses, use the wrappers under
`pipeline/fig3_attribution/`:

```bash
# Captum for enhancer AUPRC evaluation.
bash pipeline/fig3_attribution/10_captum_enhancer.sh

# Captum for motif-discovery attribution.
bash pipeline/fig3_attribution/11_captum_motif.sh

# Run motif discovery from the Fig. 3 attribution outputs.
bash pipeline/fig3_attribution/31_run_gimmemotifs.sh
bash pipeline/fig3_attribution/32_run_tfmodisco.sh

# Summarize motif outputs for the manuscript panels.
bash pipeline/fig3_attribution/33_motif_summary.sh
```

The Captum wrappers generate task lists, submit resume-aware SLURM arrays, and
submit dependent peak-calling jobs. By default they run
`MartinRufino2025_mixscape_exnp_train` with `alphagenome`; pass study, model, or
`--suffix` if rerunning a different trained model.

### Resume-Safe Attribution Analysis Route for Fig. 4 and Fig. 5

For the array-based route used in lineage analyses:

```bash
# Captum attribution and peak calling.
bash pipeline/fig4_lineage/10_captum_array.sh \
  NormanWeissman2019_filtered_mixscape_exnp_train alphagenome

# Motif discovery arrays.
bash pipeline/fig4_lineage/20_gimmemotifs_array.sh \
  NormanWeissman2019_filtered_mixscape_exnp_train alphagenome
bash pipeline/fig4_lineage/22_tfmodisco_array.sh \
  NormanWeissman2019_filtered_mixscape_exnp_train alphagenome

# Aggregate motif matrices after array jobs finish.
bash pipeline/fig4_lineage/21_gimmemotifs_aggregate.sh
bash pipeline/fig4_lineage/23_tfmodisco_aggregate.sh
```

This route writes task completion markers under `attribution_analysis/done*/`
and can be resubmitted after partial failures.

Fig. 5 uses the same resume-safe `attribution_analysis/` route, but the input
study is changed from Norman lineage perturbations to the JialongJiang CD8T
compound dataset. In practice, rerun the Captum and TF-MoDISco array wrappers
with `JialongJiang2024_CD8T_train alphagenome`.

## 6. Figure-Specific Final Steps

The sections above cover the shared preprocessing, training, and attribution
routes. The final manuscript panels for Fig. 3, Fig. 4, and Fig. 5 also require
the figure-specific post-processing wrappers below. Treat the matching
dependency document under `manuscript/method/` as the source of truth before a
full rerun.

### Fig. 3 Enhancer, Motif, and Mutation Panels

Fig. 3 uses the root-output `attribution/` route, not the
`attribution_analysis/` route. After the Fig. 3 Captum, CRE, GimmeMotifs, and
TF-MoDISco steps above, run the remaining paper-panel steps as needed:

Before running the mutation array wrapper, confirm that its GPU partition and
GRES string match the available cluster. If jobs must be pinned to selected
nodes, pass a comma-separated `MUTATION_NODES` value; otherwise the scheduler
selects nodes.

```bash
# Martin ATAC AUPRC for Fig. 3d and Fig. S9.
bash pipeline/fig3_attribution/20_enhancer_auprc_atac_martin.sh

# Seqlet metadata shared by the Fig. 3 mutation analyses.
bash pipeline/fig3_attribution/40_prepare_seqlet_metadata.sh

# Full seqlet set for Fig. 3a-c.
bash pipeline/fig3_attribution/41_prepare_mutation_targets.sh Martin_full
MUTATION_NODES=gpu-a,gpu-b bash pipeline/fig3_attribution/42_run_mutation_array.sh Martin_full
bash pipeline/fig3_attribution/43_post_mutation_figures.sh Martin_full

# Matched seqlet set for Fig. 3f.
bash pipeline/fig3_attribution/41_prepare_mutation_targets.sh Martin_matched
bash pipeline/fig3_attribution/42_run_mutation_array.sh Martin_matched
bash pipeline/fig3_attribution/43_post_mutation_figures.sh Martin_matched

# All-gene attribution-axis controls for Fig. S8.
bash pipeline/fig3_attribution/50_attribution_axis_allgenes.sh
```

Fig. 3a-c also depend on union-gene attribution
(`{pert}_union_raw_attribution.h5`), which is still generated through the
legacy dispatcher:

```bash
bash pipeline/01_slurm.sh -p captum_union
```

See `manuscript/method/figure3_dependencies.md` for the exact study keys,
matched versus full mutation-study keys, and composite figure inputs.

### Fig. 4 Lineage Panels

Fig. 4 uses the resume-safe `attribution_analysis/` route. After Captum and
motif matrix aggregation, run the lineage-specific summaries:

```bash
# Signature-axis panels for Fig. 4b-c and Fig. S11a-b.
bash pipeline/fig4_lineage/30_signature_axis.sh

# Seqlet mutation and cancellation panels for Fig. 4d and Fig. S11c.
bash pipeline/fig4_lineage/31_seqlet_mutation_array.sh
bash pipeline/fig4_lineage/32_seqlet_cancellation.sh

# SCENIC plus master-regulator UpSet panels for Fig. S12.
bash pipeline/fig4_lineage/40_scenic_master_regulator.sh
```

Fig. 4a is produced by the first step of `30_signature_axis.sh`
(`scripts/immune_differentiation/01_gene_signature.py`).

See `manuscript/method/figure4_dependencies.md` for the Norman study key,
required TF-MoDISco `pos`/`neg` matrices, seqlet metadata requirements, and
SCENIC inputs.

### Fig. 5 Drug-Mechanism Panels

Fig. 5 reuses the `attribution_analysis/` TF-MoDISco matrices for
`JialongJiang2024_CD8T_train`. Prepare the upstream study-specific outputs first:

```bash
STUDIES_OVERRIDE="JialongJiang2024_CD8T_train" \
bash pipeline/01_training/01a_transfer.sh alphagenome

bash pipeline/fig2_expression/24_clustering_signature.sh
bash pipeline/fig4_lineage/10_captum_array.sh JialongJiang2024_CD8T_train alphagenome
bash pipeline/fig4_lineage/22_tfmodisco_array.sh JialongJiang2024_CD8T_train alphagenome
bash pipeline/fig4_lineage/23_tfmodisco_aggregate.sh
```

Then run the local drug-mechanism wrapper:

```bash
bash pipeline/fig5_drug/51_drug_mechanism.sh
```

The wrapper runs the PubChem glucocorticoid screen, NR3C1 rank-percentile
panels, and candidate compound motif-vs-TF scatter plots. See
`manuscript/method/figure5_dependencies.md` for the exact CD8T study tag,
required matrices, output panel paths, and composite figure script.

## Notes

- Use `pipeline/` wrappers as the interface for rerunning analyses.
- Keep generated outputs, large input data, checkpoints, and rendered figures
  outside version control unless explicitly needed for a release.
- Before rerunning a manuscript figure, check the matching dependency document
  under `manuscript/method/` and confirm whether the figure uses `attribution/`
  or `attribution_analysis/`.

## Associated Paper

This repository accompanies the paper available at
https://doi.org/10.64898/2026.07.01.735806.
