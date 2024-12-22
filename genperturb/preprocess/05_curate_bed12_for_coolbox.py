import pandas as pd


bed12 = pd.read_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.bed", header=None, sep="\t")

tss_path = "fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.tss.bed"
tss = pd.read_csv(tss_path, sep="\t", usecols=[0,1,2,3], names=["chr", "start", "end", "gene"])

genes = list(set(tss["gene"]))

bed12["gene"] = [i.rsplit("-", 1)[0] for i in bed12[3]]
bed12["transcript"] = [i.rsplit("-", 1)[1] for i in bed12[3]]

strand_plus = pd.merge(bed12, tss, left_on=[0,1, "gene"], right_on=["chr", "start", "gene"])
strand_minus = pd.merge(bed12, tss, left_on=[0,2, "gene"], right_on=["chr", "end", "gene"])

curated_bed12 = pd.concat([strand_plus, strand_minus])

genes2 = list(set(curated_bed12["gene"]))
not_gene = tss.query('gene not in @genes2')
genes3 = list(set(not_gene["gene"]))
remained = bed12.query('gene in @genes3')

curated_bed12 = pd.concat([curated_bed12, remained])


df_sorted = curated_bed12.sort_values(by=['gene', 'transcript'], ascending=True)
df_deduped = df_sorted.drop_duplicates(subset=['chr', 'gene'], keep='first')

final_bed = df_deduped.sort_values([0,1,2]).iloc[:,:12]

final_bed[3] = [i.rsplit("-", 1)[0] for i in final_bed[3]]
final_bed.to_csv("fasta/gencode.v32.chr_patch_hapl_scaff.basic.annotation.bed12.unique.bed", sep="\t", index=False, header=False)







