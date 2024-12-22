## dataset ##
# http://projects.sanderlab.org/scperturb/
cd data/adata/
nohup wget https://zenodo.org/record/7041849/files/NormanWeissman2019_filtered.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_essential.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_rpe1.h5ad &
#https://github.com/theislab/sc-pert
nohup wget https://ndownloader.figshare.com/files/33979517/Srivatsan_2019_raw.h5ad &

## TF atlas
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217460 
nohup wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217460/suppl/GSE217460%5F210322%5FTFAtlas.h5ad.gz &
nohup wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217460/suppl/GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%5Fraw.h5ad.gz &
nohup wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217215/suppl/GSE217215_201218_ATAC.h5ad.gz &
nohup wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217066/suppl/GSE217066_210715_combinatorial.h5ad.gz &
nohup wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217066/suppl/GSE217066_210715_combinatorial_subsample.h5ad.gz &

## DSPIN
https://data.caltech.edu/records/2cjss-wgh69
nohup wget https://data.caltech.edu/records/2cjss-wgh69/files/drug_profiling_raw_counts.h5ad &
nohup wget https://data.caltech.edu/records/2cjss-wgh69/files/dosage_combination_raw_counts.h5ad &

## fasta, gff, bed ##
# https://www.gencodegenes.org/ (v32) Basic gene annotation (GRCh38.p13)
cd fasta/
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.chr_patch_hapl_scaff.basic.annotation.gff3.gz &
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.chr_patch_hapl_scaff.basic.annotation.gtf.gz &
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/GRCh38.p13.genome.fa.gz &
gunzip GRCh38.p13.genome.fa.gz
gunzip gencode.v32.chr_patch_hapl_scaff.basic.annotation.gff3.gz
gunzip gencode.v32.chr_patch_hapl_scaff.basic.annotation.gtf.gz

#https://github.com/jacobbierstedt/gfftobed
wget https://github.com/jacobbierstedt/gfftobed/archive/refs/heads/main.zip
unzip main.zip
cd gfftobed-main/
make

gfftobed-main/gfftobed -t -a gene_name fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.gff3 > fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.all_tss.bed
gfftobed-main/gfftobed -t -a gene_type fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.gff3 > fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.all_gene_type.bed

conda activate coolbox
paftools.js gff2bed -s fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.gtf | \
awk 'BEGIN { FS = "\t"; OFS = "\t" } { sub(/\|.*$/, "", $4); print }' > fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.bed
python dnaperturb/preprocess/06_curate_bed12.py


## motif ##
mkdir reference/jaspar
cd reference/jaspar
wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
grep MOTIF JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt | cut -d" " -f2- > JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme_list.txt

wget https://jaspar.elixir.no/static/clustering/2024/vertebrates/CORE/interactive_trees/clusters.tab
#wget https://jaspar.elixir.no/static/clustering/2024/vertebrates/CORE/radial_trees/annotation_table.txt

## refenrence ##
# http://humantfs.ccbr.utoronto.ca/download.php
nohup wget http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt &


## CAGE bigwig ##
curl -o data/CNhs11760.bw https://storage.googleapis.com/basenji_tutorial_data/CNhs11760.bw


## bassenji tfrecord ##
conda activate bassenji

for i in {0..7}; do
  gcloud --billing-project=ordinal-torch-413611 storage cp gs://basenji_barnyard/data/human/tfrecords/test-0-$i.tfr .
done

for i in {0..132}; do
  gcloud --billing-project=ordinal-torch-413611 storage cp gs://basenji_barnyard/data/human/tfrecords/train-0-$i.tfr .
done

for i in {0..8}; do gcloud --billing-project=ordinal-torch-413611 storage cp gs://basenji_barnyard/data/human/tfrecords/valid-0-$i.tfr . ;done



# ucsc motif
wget https://jaspar.genereg.net/download/data/2022/TFFM_table.csv

SAVE_DIR="reference/ucsc"
SAVE_DIR="."
BASE_URL="http://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022/hg38/"

for i in `grep -w -f focused_tf.txt $SAVE_DIR/TFFM_table.csv |cut -d"," -f4 | tr "," "." | sort | uniq`; do 
  for j in 1 2 3;do
    wget -P $SAVE_DIR http://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022/hg38/${i}.${j}.tsv.gz
  done
done

for i in `ls *tsv.gz`;do echo "${i}" `zcat $i | head -n1 | cut -f4 |tr '[:lower:]' '[:upper:]'` ;done > motif_list.txt

EQTL="../CL_Mono_conditional_eQTL_FDR0.05.txt"
for EQTL in `ls ../*FDR0.05.txt`;do
    EQTL_NAME=`basename $EQTL .txt`
    paste <(cut -f7-9 $EQTL) <(cut -f2,6 $EQTL) <(cut -f11- $EQTL) | tail -n +2 > ${EQTL_NAME}.bed
    BED_LIST=`for i in $(cut -d" " -f1 ../../ucsc/motif_list.txt |tr "\n" " ");do echo ../../ucsc/${i} ; done`
    NAME_LIST=`for i in $(cut -d" " -f2 ../../ucsc/motif_list.txt |tr "\n" " ");do echo ${i} ; done`
    bedtools intersect -wa -wb -a ${EQTL_NAME}.bed -b $BED_LIST -names $NAME_LIST -header > ${EQTL_NAME}_motif_intersect.bed
done


