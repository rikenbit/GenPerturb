# GenPerturb

## Overview
This repository hosts the code and tools developed for benchmarking transfer learning in genomic DNA models. The repository includes implementations for transfer learning using genomic DNA models, such as Enformer, and evaluating their ability to predict perturbation-induced gene expression changes from DNA sequences. It also provides scripts for preprocessing datasets, running experiments across multiple conditions and datasets, and applying model interpretation methods to investigate regulatory elements and transcription factor motifs. The code is designed to facilitate the reproduction of results, the exploration of transfer learning approaches, and further development of predictive models in genomic research.


## Setup Instructions

Create a virtual environment using miniconda (recommended):
```
conda create -n enformer  python=3.10
conda activate enformer
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -c bioconda bedtools
pip install pyBigWig pybedtools
conda install -c conda-forge cxx-compiler
pip install enformer-pytorch pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib
```
This environment supports transfer learning for all models available on Hugging Face, not limited to Enformer (including HyenaDNA and Nucleotide Transformer).


## Usage
The main script can be executed as follows:
```

```

Results will be saved in the results/ directory.


## Model evaluation
The scripts used to evaluate the models in the paper are listed below. Each script is categorized into those for setting up the analysis environment and those for executing the analysis.


## Citation
```
@article{TBD,
  title={Transfer learning usingon generative pretrained genomic DNA models for predicting perturbation-induced changes in gene expression},
  author={Takuya Shiihashi, Itoshi Nikaido et al.},
  year={2024}
}
```
