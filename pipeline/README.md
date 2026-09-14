# Pipeline entrypoints

Run launchers from the repository root. Set `CONDA_SH`, `PARTITION_GPU`,
`PARTITION_CPU` and `GPU_GRES` for your system; shared settings are in
`pipeline/_common.sh`.

Follow [`../reproducibility/RUN_ORDER.md`](../reproducibility/RUN_ORDER.md)
for the execution order and study-specific commands. External inputs and
generated file paths are listed in
[`../reproducibility/INPUTS_AND_OUTPUTS.md`](../reproducibility/INPUTS_AND_OUTPUTS.md).

| Directory | Analysis |
|---|---|
| `00_preprocess/` | Embeddings, single-cell preprocessing, training tables and baseline data |
| `01_training/` | Transfer, SimpleCNN, AlphaGenome folds, LoRA and full finetuning |
| `fig2_expression/` | Expression evaluation, embeddings and observed perturbation separation |
| `fig3_attribution/` | Attribution, enhancer scoring, motif discovery and mutation |
| `fig4_lineage/` | Lineage signatures, mutation and SCENIC |
| `fig5_drug/` | Compound annotations, NR3C1 ranks and motif profiles |
| `additional_analysis/` | Steps 70–75 and 78: expression uncertainty, enhancer benchmarking, paired mutation effects, tertile panels, Norman partitions, NR3C1 tests and final-size panel drawing |

Wait for submitted SLURM jobs to complete before each dependent stage.
Local analysis launchers write logs under `log/`.

## Select a study

```bash
STUDIES_OVERRIDE="NormanWeissman2019_filtered_mixscape_exnp_train" \
  bash pipeline/01_training/01a_transfer.sh alphagenome
```

To add a study, update the study/label lists in `pipeline/_common.sh` and
`dataset_model_config.py`, then run preprocessing and training for that study.
