# Conda Environments

Create an environment from the repository root:

```bash
conda env create -f environments/alphagenome.yml
```

TF-MoDISco report generation outputs only the top 10 genes by default. To relax this limitation and include more genes in the report, we apply a small modification to the processing pipeline.
For TF-MoDISco, run the documented post-install patch after environment creation:

```
conda run -n modisco bash environments/post_install/patch_modisco_report_nan_guard.sh
```

This patch also adjusts the report generation settings to prevent the output from being restricted to only the top 10 genes.

## Environment Map

| YAML | Production evidence environment | Main use |
|---|---|---|
| `singlecell.yml` | `singlecell` | single-cell preprocessing, pseudobulk construction, pySCENIC, expression-level evaluation, lightweight aggregation |
| `enformer.yml` | `enformer` | Enformer embeddings, transfer learning, Captum attribution |
| `borzoi.yml` | `borzoi` | Borzoi embeddings, transfer learning, Captum attribution |
| `alphagenome.yml` | `alphagenome` | AlphaGenome embeddings, transfer learning, Captum attribution, mutation simulation |
| `simplecnn.yml` | `bend` | from-scratch SimpleCNN baseline; the Conda environment is `simplecnn`, while the Python package import remains `bend` |
| `modisco.yml` | `modisco` | TF-MoDISco/modisco-lite motif discovery and report generation |
| `gimme.yml` | `gimme` | GimmeMotifs motif discovery and motif enrichment summaries |

