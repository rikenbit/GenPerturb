#Create individual analysis environments using conda.
#At the beginning of each script in the analysis_script directory, the specific environment to be used is indicated.

## singlecell
conda create -n singlecell  python=3.10
conda activate singlecell
pip install adpbulk pertpy muon omnipath typeguard==2.13.3

## enformer
conda create -n enformer  python=3.10
conda activate enformer
pip install pandas scikit-learn scipy matplotlib seaborn h5py
conda install -c bioconda bedtools
pip install pyBigWig pybedtools
conda install -c conda-forge cxx-compiler
pip install enformer-pytorch pytorch-lightning captum peft==0.4.0 deepspeed python-lora svglib 

## coolbox (Commands for installation without using Anaconda)
git clone https://github.com/GangCaoLab/CoolBox.git
cd CoolBox
conda create -n coolbox python=3.10.8
conda activate coolbox
conda install -c bioconda samtools tabix pairix 
pip install -r requirements.txt  
python setup.py install
conda install -c bioconda minimap2
pip install pyBigWig seaborn 

## modisco
conda create -n modisco python=3.8
conda activate modisco
conda install -c bioconda -c conda-forge bedtools meme
pip install pandas pyBigWig pybedtools modisco modisco-lite beautifulsoup4
pip install torch==2.3.1 polars pyfaidx

## deeptools
conda create -n deeptools python=3.10
conda activate deeptools
pip install deeptools
conda install -c bioconda bedtools
pip install pyBigWig pandas pybedtools
