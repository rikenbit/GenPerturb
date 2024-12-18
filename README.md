# GenPerturb
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### やること
- デモデータを作る。その他スクリプトのディレクトリを作る
- trainingだけ分けてUsageなどに書いていく。個別のデータの解析
- その他コードはpreprocessingとevaluationにまとめる。複数データをまとめているので、デモは難しいので。
  - python ---.pyで実行させる。bash系は環境変数を設定する。
  - 

This repository hosts the code and tools developed for benchmarking transfer learning in genomic DNA models. The repository includes implementations for transfer learning using genomic DNA models, such as Enformer, and evaluating their ability to predict perturbation-induced gene expression changes from DNA sequences. It also provides scripts for preprocessing datasets, running experiments across multiple conditions and datasets, and applying model interpretation methods to investigate regulatory elements and transcription factor motifs. The code is designed to facilitate the reproduction of results, the exploration of transfer learning approaches, and further development of predictive models in genomic research.

======================
## Requirements
- Feature-based : GPU memory 5GB~ 
- Fine tuning : GPU memory 49GB~ (e.g. NVIDIA RTX 6000 Ada)

## Setup Instructions
Create a virtual environment using miniconda:
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
Environment variables are configured according to the dataset and training method.
```
$ STUDY=
$ STUDY_SUFFIX=
```

Demonstrating feature-based transfer learning using example datasets.
The main script can be executed as follows:
```
python 02_qsub_script.py $STUDY $STUDY_PLAN $MODEL
```

Results will be saved in the results/ directory.

## Output


## Preprocessing of single-cell data and pre-embedding

## Model evaluation
The scripts used to evaluate the models in the paper are listed below. Each script is categorized into those for setting up the analysis environment and those for executing the analysis.
- Environment : /environment directory
- Script : /analysis directory
```
$ cd GenPerturb # All scripts should be executed within this directory.
```


### Evaluation of clustering and gene signature
- Environment :
  - /environment/
- Script :
  - /analysis/
```

```

### Model interpretation using Captum
- Environment :
  - /environment/
- Script :
  - /analysis/
For Enformer:
```
$ STUDY=
$ STUDY_SUFFIX=
python 10_captum.py $STUDY $STUDY_SUFFIX seq top
python 10_captum.py $STUDY $STUDY_SUFFIX seq all
python 10_captum.py $STUDY $STUDY_SUFFIX seq condition
python 10_captum.py $STUDY $STUDY_SUFFIX pert condition
python 10_captum.py $STUDY $STUDY_SUFFIX pert all
python 10_captum.py $STUDY $STUDY_SUFFIX pert tf
python 11_evaluate_attribution.py $STUDY $STUDY_SUFFIX
```
For HyenaDNA and Nucleotide Transformer
```
$ STUDY=
$ STUDY_SUFFIX=
$ MODEL=
python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL seq condition
python 11_evaluate_attribution.py $STUDY $STUDY_SUFFIX
```

### Visualization of attribution and ChIP-seq data in BigWig format using CoolBox.
- Environment :
  - /environment/
- Script :
  - /analysis/
 
    
### Enrichment analysis of genome tracks
- Environment :
  - /environment/
- Script :
  - /analysis/


### Motif detection using TF-MoDISco
- Environment :
  - /environment/
- Script :
  - /analysis/

## Citation
```
@article{TBD,
  title={Transfer learning usingon generative pretrained genomic DNA models for predicting perturbation-induced changes in gene expression},
  author={Takuya Shiihashi, Itoshi Nikaido et al.},
  year={2024}
}
```
