import pandas as pd



bassenji = pd.read_csv("fasta/data_human_sequences.bed", sep="\t", names=["chr", "start_x", "end_x", "training"])
bed = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.all_tss.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand"])
gene_type = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.all_gene_type.bed", sep="\t", names=["chr", "start", "end", "Gene_type", "score", "strand"])
bed["Gene_type"] =  gene_type["Gene_type"]
bed = bed[bed["chr"].isin(list(set(bassenji.chr)))]
bed = bed.query('Gene_type in ["protein_coding", "lncRNA"]')
bed = bed.drop("Gene_type", axis=1)

df = pd.DataFrame()
for i in set(bed["chr"]):
    print(i)
    #i= "chr1"
    df_tmp = pd.merge(bed.query('chr == @i'), bassenji.query('chr == @i'), on="chr")
    df_tmp = df_tmp[(df_tmp["start"] > df_tmp["start_x"]) & (df_tmp["end"] < df_tmp["end_x"])]
    df_tmp = df_tmp.drop(["start_x","end_x"], axis=1).drop_duplicates()
    df = pd.concat([df, df_tmp])

df = df.sort_values(["chr", "start"])
df["training"][df["training"] == "valid"] = "val"

df.to_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed", sep="\t", index=False, header=False)





