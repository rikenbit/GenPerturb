#!/usr/bin/env Rscript
# Convert GSE277747 (Wu et al., 2024) SingleCellExperiment (with altExp
# "ATAC") into plain mtx / tsv files that Python can read without R.
#
# Usage: Rscript 11a_convert_Wu2024_sce.R input.RDS out_dir

suppressMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(Matrix)
  library(S4Vectors)
  library(GenomicRanges)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: 11a_convert_Wu2024_sce.R input.RDS out_dir")
in_rds <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

sce <- readRDS(in_rds)
cat(sprintf("[read] %s  %dx%d\n", class(sce)[1], nrow(sce), ncol(sce)))

# ---- RNA ----
rna_m <- assay(sce, "counts")
writeMM(rna_m, file.path(out_dir, "rna.mtx"))
rd <- as.data.frame(rowData(sce))
rd$rowname <- rownames(rna_m)
write.table(rd, file.path(out_dir, "rna_features.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(colnames(rna_m), file.path(out_dir, "rna_barcodes.tsv"))
cat(sprintf("[rna ] %dx%d written\n", nrow(rna_m), ncol(rna_m)))

# ---- colData (guide assignments) ----
cd <- as.data.frame(colData(sce))
cd$barcode <- rownames(cd)
write.table(cd, file.path(out_dir, "cells.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[obs ] %d cells, cols=%s\n", nrow(cd),
            paste(colnames(cd), collapse = ",")))

# ---- ATAC altExp ----
if ("ATAC" %in% altExpNames(sce)) {
  atac <- altExp(sce, "ATAC")
  am <- assay(atac, "counts")
  writeMM(am, file.path(out_dir, "atac.mtx"))
  ard <- as.data.frame(rowData(atac))
  ard$rowname <- rownames(am)
  write.table(ard, file.path(out_dir, "atac_features.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  writeLines(colnames(am), file.path(out_dir, "atac_barcodes.tsv"))
  cat(sprintf("[atac] %dx%d written\n", nrow(am), ncol(am)))

  rr <- rowRanges(atac)
  if (length(rr) > 0) {
    gr <- unlist(rr)
    if (length(gr) == 0) {
      cat("[atac] rowRanges empty\n")
    } else {
      df <- data.frame(
        feature = rep(names(rr), lengths(rr)),
        chrom   = as.character(seqnames(gr)),
        start   = start(gr),
        end     = end(gr),
        strand  = as.character(strand(gr)),
        stringsAsFactors = FALSE
      )
      write.table(df, file.path(out_dir, "atac_peak_ranges.tsv"),
                  sep = "\t", quote = FALSE, row.names = FALSE)
      cat(sprintf("[atac] peak_ranges: %d entries\n", nrow(df)))
    }
  }
}

cat("[done]\n")
