# GenPerturb
![Python Version](https://img.shields.io/badge/python-3.8|3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)


Welcome to GenPerturb Github repository.  
<br>
This repository hosts the code and tools developed for benchmarking transfer learning in genomic DNA models. The repository includes implementations for transfer learning using genomic DNA models, such as Enformer, and evaluating their ability to predict perturbation-induced gene expression changes from DNA sequences.  
It also provides scripts for preprocessing datasets, running experiments across multiple conditions and datasets, and applying model interpretation methods to investigate regulatory elements and transcription factor motifs.  
The code is designed to facilitate the reproduction of results, the exploration of transfer learning approaches, and further development of predictive models in genomic research.  
<br>
<br>

---
## Requirements
- Feature-based : GPU memory 8GB~
- LoRA : GPU memory 20GB~
- Fine-tuning : GPU memory 49GB~ (e.g. NVIDIA RTX 6000 Ada)

## Setup Instructions
Create an environment using miniconda:
```
conda create -n enformer  python=3.10
conda activate enformer
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -c bioconda bedtools
pip install pyBigWig pybedtools
conda install -c conda-forge cxx-compiler
pip install enformer-pytorch pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib
```
Then,
```
$ git clone https://github.com/rikenbit/GenPerturb.git
$ cd GenPerturb # All scripts should be executed within this directory.
```

This environment supports transfer learning for all models available on Hugging Face, not limited to Enformer (including HyenaDNA and Nucleotide Transformer).


## Usage
Demonstrating transfer learning using Enformer and example datasets.
Environment variables are configured according to the dataset and training method.
```
$ MODEL="enformer" # To train other models, change to hyena_dna_last, hyena_dna_mean, nucleotide_transformer_tss, or nucleotide_transformer_cls.
$ STUDY="NormanWeissman2019_filtered_mixscape_exnp_train"
$ STUDY_SUFFIX="${MODEL}_transfer_epoch100_batch256_adamw5e3"
```
The demo datasets can be downloaded from the sources listed below.
- pseudo-bulk, pre-embedding, TSS bed
  - https://drive.google.com/drive/folders/1YLDwZWXUEzv2_i4OgFdOemmGtR5S9XMf?usp=sharing
- fasta file
  - https://www.gencodegenes.org/human/release_32.html
The downloaded data should be organized using the directory structure shown below.
```
$ cd GenPerturb
$ ls
data/${STUDY}.tsv # pseudo-bulk Perturb-seq data
data/${STUDY}.h5 # pre-embedding data
fasta/{STUDY}.bed # TSS positions of genes included in the training data
fasta/GRCh38.p13.genome.fa # fasta file
```

The training script for feature-based transfer learning can be executed as follows:
```
$ cp analysis_script/02_qsub_script.py .
$ python 02_qsub_script.py $STUDY transfer $MODEL # When performing fine-tuning, replace "transfer" with either "finetuning" or "lora".
```

The prediction and evaluation results will be saved in the prediction/ and figures/ directory.  
The trained checkpoint file will be saved in the logs/ directory.

---
## Preprocessing of single-cell data and pre-embedding
Data preparation for training and model evaluation is performed using scripts in the following directory.  
This includes downloading and preprocessing single-cell data, obtaining pre-embedding values, creating BED files that define TSS regions, downloading ChIP-seq and motif data, and so on.
```
genperturb/preprocess/*
```

---
## Model evaluation 
The evaluation scripts required for reproducing the results of the paper are listed below.  
Most scripts perform comparisons between datasets or models. To execute these scripts, it is necessary to prepare all the data used under the conditions specified in the paper.
```
analysis_script/*
```
The analysis environments are set up based on the scripts provided below, with each analysis script specifying the environment required for its execution.
```
crearte_env.sh
```
### Evaluation items included in the script
- Evaluation of clustering and gene signature
- Model interpretation using Captum
- Visualization of attribution and ChIP-seq data in BigWig format using CoolBox.
- Enrichment analysis of genome tracks
- Motif detection using TF-MoDISco

---
## Acknowledgments
We thank the members of the Bioinformatics Research Team in RIKEN Center for Biosystems Dynamics Research, particularly Akihiro Matsushima, for their management of the IT infrastructure.

## Citation
```
@article{TBD,
  title={Transfer learning using generative pretrained genomic DNA models for predicting perturbation-induced changes in gene expression},
  author={Takuya Shiihashi, Itoshi Nikaido et al.},
  year={2024}
}
```
