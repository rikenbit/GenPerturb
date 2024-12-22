
# https://github.com/inutano/chip-atlas/wiki
wget https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab
grep -w hg38 reference/chipatlas/experimentList.tab |grep -v -e "Input control" |cut -f1-6,8- |grep Histone| grep -v "ATAC-Seq" |sort -k4 |grep K-562|less


# K562
SRX097105      CEBPA
SRX2423912      CEBPB
SRX2423913      CEBPB
SRX2424502      FOXA1
SRX2424503      FOXA1
SRX2424509      IRF1
SRX2424510      IRF1
SRX1431733      SPI1
SRX1431734      SPI1
SRX1431735      SPI1
SRX16495812	JUN
SRX5457220	ETS2
SRX3321888	EGR1

# MOLM-13
SRX5732677      KMT2A

# U-266
SRX3070540      PRDM1

# GM17212
SRX4342285      AHR
SRX4342286      AHR

# CD4+ T cells
DRX440350       TP73

# Hep G2
SRX10475577     HNF4A
SRX10475578     HNF4A
SRX10478046     SNAI1
SRX10478047     SNAI1

# K562 Histon
SRX10184518 ATAC
SRX10184470 H3K27
SRX9243952      H3K27me3
SRX8725106      H3K36me3
SRX2636149 H3K4me3
SRX2636150 H3K4me3
SRX028593 H3K9me3
SRX038639 H3K9me3
ERX989298 CTCF

## donwload command ##
#(Threshold = 05, 10, or 20)
sample_ids=(
"SRX097105"
)

sample_ids=(
SRX2423912
SRX2424502
SRX2424509
SRX1431734
SRX5732677
SRX3070540
SRX4342285
DRX440350
SRX10475577
SRX10478046
SRX9243952
SRX8725106
SRX2636149
SRX028593
)

sample_ids=(
SRX16495812
SRX5457220
SRX3321888
)

for sample_id in "${sample_ids[@]}";do
    bw_file="${sample_id}.bw"
    bed_file="${sample_id}.05.bed"
    nohup wget "https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/${bw_file}" &
    nohup wget "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/${bed_file}" &
done

for sample_id in "${sample_ids[@]}";do
    bed_file="${sample_id}.05.bed"
    bed6_file="${sample_id}.05.bed6.bed"
    cut -f-6 "$bed_file" > "$bed6_file"
done





## download Hi-C ##
#https://data.4dnucleome.org/experiment-set-replicates/4DNESI7DEJTM/
# download in webpage with account login

curl -O -L --user B4LJK6KF:4335p5n4heuj3k74 https://data.4dnucleome.org/files-processed/4DNFI18UHVRO/@@download/4DNFI18UHVRO.mcool



