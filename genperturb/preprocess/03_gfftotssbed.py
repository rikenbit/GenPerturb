#!/usr/bin/env python3
# -*- coding: utf-8 -*-

IN_GFF3  = "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.gff3"
OUT_BED  = "fasta/gencode.v46.chr_patch_hapl_scaff.basic.annotation.all_tss.bed"

KEEP_TX_TYPES   = {"protein_coding", "lncRNA"}  # transcript_type
KEEP_GENE_TYPES = {"protein_coding", "lncRNA"}  # gene_type
PRIMARY_CHROMS  = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

def get_attr(attrs: str, key: str) -> str:
    needle = key + "="
    for part in attrs.split(";"):
        part = part.strip()
        if part.startswith(needle):
            return part[len(needle):]
    return ""

def get_tag(attrs: str) -> str:
    return get_attr(attrs, "tag") 

def get_rank(attrs: str) -> int:
    for token in attrs.replace(";", "|").split("|"):
        token = token.strip()
        if token.startswith("rank") and token[4:].isdigit():
            return int(token[4:])
    return 999999

print(f"[INFO] Input : {IN_GFF3}")
print(f"[INFO] Output: {OUT_BED}")

gene_id_to_symbol = {}
gene_id_to_type   = {}

with open(IN_GFF3, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line or line[0] == "#":
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 9:
            continue
        chrom, source, feature, start, end, score, strand, phase, attrs = cols
        if feature != "gene":
            continue

        gene_id = get_attr(attrs, "gene_id") or get_attr(attrs, "ID")
        if not gene_id:
            continue

        gene_type = get_attr(attrs, "gene_type") or get_attr(attrs, "gene_biotype") or ""
        symbol    = get_attr(attrs, "gene_name") or ""

        gene_id_to_type[gene_id] = gene_type
        if symbol:
            gene_id_to_symbol[gene_id] = symbol

n_tx = 0
n_primary = 0
n_basic = 0
n_keep_type = 0
n_with_symbol = 0

# gene_id -> (score_tuple, (chrom, bed_start, bed_end, symbol, strand))
best_by_gene = {}

with open(IN_GFF3, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line or line[0] == "#":
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 9:
            continue

        chrom, source, feature, start, end, score, strand, phase, attrs = cols
        if feature != "transcript":
            continue
        n_tx += 1

        if chrom not in PRIMARY_CHROMS:
            continue
        n_primary += 1

        tag = get_tag(attrs)
        if "basic" not in tag:
            continue
        n_basic += 1

        tx_type = get_attr(attrs, "transcript_type") or ""
        if tx_type not in KEEP_TX_TYPES:
            continue

        gene_id = get_attr(attrs, "gene_id")
        if not gene_id:
            continue
        gene_type = gene_id_to_type.get(gene_id, "")
        if gene_type and gene_type not in KEEP_GENE_TYPES:
            continue
        n_keep_type += 1

        symbol = gene_id_to_symbol.get(gene_id, "")
        if not symbol or symbol.startswith("ENSG"):
            continue
        n_with_symbol += 1

        start_i = int(start)
        end_i   = int(end)

        tss_1b = start_i if strand == "+" else end_i

        tss_0b = tss_1b - 1
        bed_start = tss_0b
        bed_end   = tss_0b + 1

        tx_id = get_attr(attrs, "transcript_id") or get_attr(attrs, "ID") or ""

        is_canonical = 0 if "Ensembl_canonical" in tag else 1
        rank = get_rank(attrs)
        score_tuple = (is_canonical, rank, tx_id)

        prev = best_by_gene.get(gene_id)
        if prev is None or score_tuple < prev[0]:
            best_by_gene[gene_id] = (score_tuple, (chrom, bed_start, bed_end, symbol, strand))

with open(OUT_BED, "w", encoding="utf-8") as out:
    for gene_id, (_score, rec) in best_by_gene.items():
        chrom, bed_start, bed_end, symbol, strand = rec
        out.write(f"{chrom}\t{bed_start}\t{bed_end}\t{symbol}\t0\t{strand}\n")

print(f"[INFO] transcripts total                : {n_tx}")
print(f"[INFO] transcripts on primary chroms   : {n_primary}")
print(f"[INFO] transcripts with basic          : {n_basic}")
print(f"[INFO] transcripts keep gene_type      : {n_keep_type}")
print(f"[INFO] transcripts with gene_symbol    : {n_with_symbol}")
print(f"[INFO] genes with representative (final): {len(best_by_gene)}")
print("[INFO] done.")

