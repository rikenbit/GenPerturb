## dataset ##
# http://projects.sanderlab.org/scperturb/
cd data/adata/
nohup wget https://zenodo.org/record/7041849/files/NormanWeissman2019_filtered.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_essential.h5ad &
nohup wget https://zenodo.org/record/7041849/files/ReplogleWeissman2022_rpe1.h5ad &
#https://github.com/theislab/sc-pert
nohup wget https://ndownloader.figshare.com/files/33979517/Srivatsan_2019_raw.h5ad &

## DSPIN
https://data.caltech.edu/records/2cjss-wgh69
nohup wget https://data.caltech.edu/records/2cjss-wgh69/files/drug_profiling_raw_counts.h5ad &
nohup wget https://data.caltech.edu/records/2cjss-wgh69/files/dosage_combination_raw_counts.h5ad &

## Multi-ome
OUTDIR="data/MartinRufino"
BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274113/suppl"
mkdir -p "$OUTDIR"

wget -c -P "$OUTDIR" "${BASE_URL}/GSE274113_annotated_metadata.csv.gz"
gunzip -kf "$OUTDIR/GSE274113_annotated_metadata.csv.gz"

for rep in 1 2 3 4 5 6 7 8 9 10 12 13 14 16; do
    echo "  Downloading rep${rep}..."
    wget -c -P "$OUTDIR" "${BASE_URL}/GSE274113_rep${rep}_filtered_feature_bc_matrix.h5"
done

for num in 1 2 3 4 5 6 7 8 9 10 12 13 14 16; do
    echo "  Downloading matrix ${num}..."
    wget -c -P "$OUTDIR" "${BASE_URL}/GSE274113_filtered_feature_bc_matrix_${num}.h5"
done


## Multi-ome: Wu et al. 2024 (GSE277747, MultiPerturb-seq) ##
# One RDS-packed SingleCellExperiment (rna + altExp "ATAC" + per-cell guide
# assignments). After download extract the .RDS and run
# genperturb/preprocess/11a_convert_Wu2024_sce.R to emit mtx/tsv/cells.tsv.
OUTDIR="data/GSE277747"
mkdir -p "$OUTDIR"
wget -c -P "$OUTDIR" "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE277nnn/GSE277747/suppl/GSE277747_RAW.tar"


## Multi-ome: Shevade/Yang et al. 2025 (GSE288996, CAT-ATAC, K562 DMSO only) ##
# Per-cell gRNA is not published; only per-sample aggregate guide counts are.
# For GenPerturb we download just the K562 DMSO multimodal (RNA+Peaks) mtx
# trios and the per-sample guide summaries. ATAC fragments (~5 GB each) are
# skipped here — pass --with-fragments to 11a_build_Shevade2025_h5ad.py if
# they're fetched later.
OUTDIR="data/GSE288996"
mkdir -p "$OUTDIR"
SAMPLE_BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8780nnn"

# K562 DMSO RNA rep1 (GSM8780566)
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780566/suppl/GSM8780566_K562_DMSO_RNA_1_barcodes.tsv.gz"
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780566/suppl/GSM8780566_K562_DMSO_RNA_1_features.tsv.gz"
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780566/suppl/GSM8780566_K562_DMSO_RNA_1_matrix.mtx.gz"
# K562 DMSO RNA rep2 (GSM8780567)
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780567/suppl/GSM8780567_K562_DMSO_RNA_2_barcodes.tsv.gz"
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780567/suppl/GSM8780567_K562_DMSO_RNA_2_features.tsv.gz"
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780567/suppl/GSM8780567_K562_DMSO_RNA_2_matrix.mtx.gz"
# K562 DMSO per-sample guide UMIs (tiny txt.gz)
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780562/suppl/GSM8780562_K562_DMSO_guideRNA_1.txt.gz"
wget -c -P "$OUTDIR" "${SAMPLE_BASE}/GSM8780563/suppl/GSM8780563_K562_DMSO_guideRNA_2.txt.gz"


## fasta, gff, bed ##
# https://www.gencodegenes.org/ (v46) Basic gene annotation (GRCh38.p14)
cd fasta/
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.chr_patch_hapl_scaff.basic.annotation.gff3.gz &
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.chr_patch_hapl_scaff.basic.annotation.gtf.gz &
nohup wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/GRCh38.p14.genome.fa.gz &
gunzip GRCh38.p14.genome.fa.gz
gunzip gencode.v46.chr_patch_hapl_scaff.basic.annotation.gff3.gz
gunzip gencode.v46.chr_patch_hapl_scaff.basic.annotation.gtf.gz

python genperturb/preprocess/03_gfftotssbed.py 



## Enhancer score
cd data
## ABC score
wget ftp://ftp.broadinstitute.org/outgoing/lincRNA/ABC/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz

## rE2G K562
wget https://www.encodeproject.org/files/ENCFF497HEA/@@download/ENCFF497HEA.bed.gz # threshold
wget https://www.encodeproject.org/files/ENCFF246ZQE/@@download/ENCFF246ZQE.bed.gz # full

## rE2G K562 (extended)
wget https://www.encodeproject.org/files/ENCFF269DKY/@@download/ENCFF269DKY.bed.gz # threshold
wget https://www.encodeproject.org/files/ENCFF950FTI/@@download/ENCFF950FTI.bed.gz # full


## gimmemotifs reference
mkdir -p reference/gimmemotifs/motif_db
cd reference/gimmemotifs/motif_db
wget https://raw.githubusercontent.com/vanheeringen-lab/gimmemotifs/master/data/motif_databases/JASPAR2022_vertebrates.pfm


## motif ##
mkdir reference/jaspar
cd reference/jaspar
wget https://jaspar.elixir.no/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
grep MOTIF JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt | cut -d" " -f2- > JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme_list.txt

wget https://jaspar.elixir.no/static/clustering/2022/vertebrates/CORE/interactive_trees/clusters.tab
#wget https://jaspar.elixir.no/static/clustering/2024/vertebrates/CORE/radial_trees/annotation_table.txt

## refenrence ##
# http://humantfs.ccbr.utoronto.ca/download.php
nohup wget http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt &


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



