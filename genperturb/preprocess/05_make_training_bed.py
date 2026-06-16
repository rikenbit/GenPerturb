import pandas as pd


alphagenome = pd.read_csv("fasta/alphagenome/all_regions.bed", sep="\t", names=["chr", "start_x", "end_x", "fold"])
bed = pd.read_csv("fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.all_tss.bed", sep="\t", names=["chr", "start", "end", "Gene", "score", "strand"])
bed = bed[bed["chr"].isin(list(set(alphagenome.chr)))]

df = pd.DataFrame()
for i in set(bed["chr"]):
    print(i)
    df_tmp = pd.merge(bed.query('chr == @i'), alphagenome.query('chr == @i'), on="chr")
    df_tmp = df_tmp[(df_tmp["start"] > df_tmp["start_x"]) & (df_tmp["end"] < df_tmp["end_x"])]
    df_tmp = df_tmp.drop(["start_x","end_x"], axis=1).drop_duplicates()
    df = pd.concat([df, df_tmp])

df = df.sort_values(["chr", "start"])

df.to_csv("fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.tss.bed", sep="\t", index=False, header=False)
