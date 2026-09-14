# Conda environments

Create each required environment from the repository root:

```bash
conda env create -f environments/singlecell.yml
conda env create -f environments/alphagenome.yml
conda env create -f environments/modisco.yml
conda env create -f environments/gimme.yml
```

| YAML | Environment | Use |
|---|---|---|
| `singlecell.yml` | `singlecell` | Single-cell preprocessing, pseudobulk, expression evaluation, pySCENIC and summary analyses |
| `alphagenome.yml` | `alphagenome` | AlphaGenome embeddings, fitting, attribution and mutation |
| `borzoi.yml` | `borzoi` | Borzoi embeddings, fitting and attribution |
| `enformer.yml` | `enformer` | Enformer embeddings, fitting and attribution |
| `simplecnn.yml` | `simplecnn` | From-scratch CNN baseline (Python package `bend`) |
| `modisco.yml` | `modisco` | TF-MoDISco motif discovery and reports |
| `gimme.yml` | `gimme` | GimmeMotifs discovery and enrichment |

Apply the TF-MoDISco report patch after environment creation:

```bash
conda run -n modisco bash environments/post_install/patch_modisco_report_nan_guard.sh
```

Register the GRCh38 FASTA before running GimmeMotifs:

```bash
conda activate gimme
genomepy install -p local fasta/GRCh38.p14.genome.fa
```
