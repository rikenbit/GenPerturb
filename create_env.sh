# Create individual analysis environments using conda.
# Adjust PROJECT_ROOT, BEND_SRC_DIR, CUDA/PyTorch versions, and token setup for
# the target system before running these blocks.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

## singlecell
conda create -n singlecell  python=3.10
conda activate singlecell
pip install adpbulk pertpy muon omnipath typeguard==2.13.3 decoupler==1.6.0
pip install pyscenic

## enformer
conda create -n enformer  python=3.10
conda activate enformer
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -y -c bioconda bedtools
pip install pyBigWig pybedtools
conda install -y -c conda-forge cxx-compiler
pip install enformer-pytorch pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib 
pip install macs3

## borzoi
conda create -y -n borzoi python=3.10
conda activate borzoi
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -y -c bioconda bedtools
pip install pyBigWig pybedtools
conda install -c nvidia/label/cuda-12.6.0 cuda-toolki=12.1t
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
FLASH_VER=2.8.0.post2
WHL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_VER}/flash_attn-${FLASH_VER}+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir "${WHL_URL}"
pip install borzoi-pytorch polars pyfaidx pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib

## alphagenome
conda create -y -n alphagenome python=3.11
conda activate alphagenome
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -y -c bioconda bedtools
pip install pyBigWig pybedtools macs3 openpyxl
pip install alphagenome-pytorch pytorch-lightning captum polars peft==0.4.0 deepspeed python-lora git+https://github.com/google-deepmind/alphagenome_research.git
cp .env.template .env # write HF_TOKEN to .env

## simple cnn
conda create -y -n bend python=3.10
conda activate bend
BEND_SRC_DIR="${BEND_SRC_DIR:-${PROJECT_ROOT}/external/BEND}"
mkdir -p "$(dirname "${BEND_SRC_DIR}")"
git clone https://github.com/frederikkemarin/BEND.git "${BEND_SRC_DIR}"
cd "${BEND_SRC_DIR}"
git checkout ac6e80c75e09d83cf47a7b4bcf0e44599c5706cf
pip install -r requirements.txt
pip install -e .
pip install scipy matplotlib seaborn h5py polars pyfaidx \
  pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib

## modisco
conda create -y -n modisco python=3.11
conda activate modisco
conda install -y -c bioconda -c conda-forge bedtools meme
pip install pandas pyBigWig pybedtools modisco beautifulsoup4
pip install torch polars pyfaidx

### fix error
#Prevent `AttributeError` during report generation when TomTom match entries contain `NaN` values instead of strings by checking the value type before calling `.strip()`.
#~/miniconda/envs/modisco/lib/python3.11/site-packages/modiscolite/descriptive_report.py
#      406          tomtom_logos[pattern_tag] = {}                                                                                                               
#      407          for i in range(top_n_matches):                                                                                                               
#      408              match_key = f'match_{i}'                                                                                                                 
#      409 -            if match_key in matches and matches[match_key]:                                                                                   
#      409 +            if match_key in matches and isinstance(matches[match_key], str) and matches[match_key]:                                           
#      410                  match_name = matches[match_key].strip()
#      411                  if match_name in motifs:
#      412                      # Create logo for this match
#     ...
#      434          name_parts = []
#      435          for i in range(min(top_n_matches, 3)):  # Use max 3 matches for name
#      436              match_key = f'match_{i}'
#      437 -            if match_key in matches and matches[match_key]:                                                                                   
#      437 +            if match_key in matches and isinstance(matches[match_key], str) and matches[match_key]:                                           
#      438                  match_name = matches[match_key].strip()
#      439                  # Take first 10 characters
#      440                  name_parts.append(match_name[:10])



## gimmemotifs
git clone https://github.com/vanheeringen-lab/gimmemotifs.git gimmemotifs_src
cd gimmemotifs_scr
git checkout develop
conda env create -f requirements.yaml
conda activate gimme
python setup.py build  # installs the motif discovery tools
pip install -e .       # installs gimmemotifs (in editable mode)
genomepy install -p local  "${PROJECT_ROOT}/fasta/GRCh38.p14.genome"
