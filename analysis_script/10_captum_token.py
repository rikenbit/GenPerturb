import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
import pyBigWig
from pybedtools import BedTool
import h5py
from scipy.stats import zscore
import os
import re
import sys
from captum.attr import Saliency, DeepLift, IntegratedGradients, InputXGradient, LayerGradientXActivation
from genperturb.model._genperturb import GenPerturb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
study_name = sys.argv[1]
df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
bed   = f'fasta/{study_name}.bed'
fasta = f'fasta/GRCh38.p13.genome.fa'

pretraind_model = sys.argv[3]
#pretraind_model = "hyena_dna_tss"
#pretraind_model = "nucleotide_transformer_tss"
if pretraind_model in ["hyena_dna_tss", "hyena_dna_last"]:
    from transformers import AutoTokenizer
    from genperturb.dataloaders._genome import GenomeIntervalDataset
    context_length = 159_998
    hdf5  = f'data/{study_name}_hyena_tss.h5'
    checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ds = GenomeIntervalDataset(bed_file=bed, fasta_file=fasta, return_sequence=True, context_length=context_length)
elif pretraind_model in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
    from transformers import AutoTokenizer
    from genperturb.dataloaders._genome import GenomeIntervalDataset
    context_length = 12_282
    hdf5  = f'data/{study_name}_nt_tss.h5'
    checkpoint = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ds = GenomeIntervalDataset(bed_file=bed, fasta_file=fasta, return_sequence=True, context_length=context_length)

if pretraind_model == "hyena_dna_tss":
    hdf5 = f'data/{study_name}_hyena_tss.h5'
    emb_method = 'tss'
elif pretraind_model == "hyena_dna_last":
    hdf5 = f'data/{study_name}_hyena_last.h5'
    emb_method = 'last'
elif pretraind_model == "nucleotide_transformer_tss":
    hdf5 = f'data/{study_name}_nt_tss.h5'
    emb_method = 'tss'
elif pretraind_model == "nucleotide_transformer_cls":
    hdf5 = f'data/{study_name}_nt_cls.h5'
    emb_method = 'cls'


study_suffix = sys.argv[2]
#study_suffix = f"{pretraind_model}_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'

model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained=pretraind_model, training_method="transfer", study=study, emb_method=emb_method)
model.load_model()
model.load_pretrained_model()
model = model.to(device)

pred = np.load(f"prediction/{study}/prediction.npy")
df2 = pd.DataFrame(pred)
df2.columns = df.columns[1:]
df2.index = df.index

ctrl = df2.columns[0]
df3 = (df2.T - df2[ctrl]).T.drop(ctrl, axis=1).copy()



def cal_attribution(model, ds, pert_num, gene_num, context_length=159_998):
    # https://captum.ai/tutorials/IMDB_TorchText_Interpret
    print("GradientXActivation")
    if pretraind_model in ["hyena_dna_tss", "hyena_dna_last"]:
        sequence = ds[gene_num]
        input_seq = tokenizer(sequence, return_tensors="pt", truncation=True)["input_ids"] #.unsqueeze(0)
        ixg = LayerGradientXActivation(model, model.module.pretrained_model.hyena.backbone.embeddings.word_embeddings)
        attributions_ixg = ixg.attribute(input_seq.to(device), target=pert_num)
        attributions_ixg = attributions_ixg.sum(dim=2).squeeze(0)
        attributions_ixg = attributions_ixg[:context_length]
        attributions_ixg = torch.cat((torch.zeros(1), attributions_ixg.cpu(), torch.zeros(1))) # padding
    elif pretraind_model in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
        sequence = ds[gene_num]
        input_seq = tokenizer(sequence, return_tensors="pt", truncation=True)["input_ids"] #.unsqueeze(0)
        ixg = LayerGradientXActivation(model, model.module.pretrained_model.esm.embeddings.word_embeddings)
        attributions_ixg = ixg.attribute(input_seq.to(device), target=pert_num)
        attributions_ixg = attributions_ixg.sum(dim=2).squeeze(0)
        attributions_ixg = torch.repeat_interleave(attributions_ixg, repeats=6)
        attributions_ixg = attributions_ixg[6:]
        attributions_ixg = torch.cat((torch.zeros(6), attributions_ixg.cpu())) # padding
    return attributions_ixg.cpu()

def cal_mean_attribution(attribution, output="", nbin=128):
    conv = F.avg_pool1d(attribution.unsqueeze(0).unsqueeze(0), kernel_size=nbin, stride=nbin, padding=0)
    attribution_mean = conv[0, 0, :].detach().numpy()
    return attribution_mean

def generate_bed_dataframe(chromosome, start, end, bin_size=128):
    data = []
    for i in range(start, end, bin_size):
        bin_start = i
        bin_end = min(i + bin_size, end)
        data.append([chromosome, bin_start, bin_end])
    return pd.DataFrame(data, columns=['chromosome', 'start', 'end'])

def get_peaks(attributions_mean, chromosome, seq_start, seq_end, gene, pert, zscore_threshold=2):
    attribution_zscore = zscore(attributions_mean)
    df_bed = generate_bed_dataframe(chromosome, int(seq_start), int(seq_end))
    df_bed["gene"] = gene
    df_bed["score"] = "."
    df_bed["strand"] = "."
    df_bed["pert"] = pert
    df_bed["attribution"] = attributions_mean
    peak_flag = np.where(attribution_zscore >= zscore_threshold, 1, np.where(attribution_zscore <= -zscore_threshold, -1, 0))
    df_bed["peak"] = peak_flag
    return df_bed, peak_flag


### captum ###
torch.manual_seed(123)
np.random.seed(123) 

def ctrl_process(gene, attribution="ixg"):
    print(gene)
    pert_num = 0
    gene_num = ds.df.with_row_count("row_number").filter(pl.col("column_4") == gene)["row_number"][0]
    print(f"pert : {pert_num}, gene : {gene_num}")
    ctrl_attributions_ixg = cal_attribution(model, ds, pert_num, gene_num, context_length=context_length)
    return ctrl_attributions_ixg

def main_process(perturbations, genes, study, suffix="", modisco=False, attribution="ixg", nbin=128, save=""):
    attributions_score = []
    attributions_score_fc = []
    peaks    = pd.DataFrame()
    peaks_fc = pd.DataFrame()
    for pert, gene in zip(perturbations, genes):
        if "ctrl_attributions_ixg" not in locals():
            ctrl_attributions_ixg = ctrl_process(gene)
        print(f"{pert} {gene}")
        pert_num = df2.columns.get_loc(pert)
        gene_num = ds.df.with_row_index("row_number").filter(pl.col("column_4") == gene)["row_number"][0]
        chromosome = ds.df[gene_num, 0]
        seq_start  = int(ds.df[gene_num, 1] - (context_length / 2))
        seq_end    = int(ds.df[gene_num, 1] + (context_length / 2))
        attributions_ixg = cal_attribution(model, ds, pert_num, gene_num, context_length=context_length)
        attributions_mean = cal_mean_attribution(attributions_ixg)

        print("----------------")
        print(attributions_ixg)
        print(attributions_ixg.shape)
        print(ctrl_attributions_ixg.shape)
        print(attributions_mean.shape)
        print("----------------")

        peak_attr, peak_flag = get_peaks(attributions_mean, chromosome, seq_start, seq_end, gene, pert)
        attributions_ixg_fc = attributions_ixg - ctrl_attributions_ixg
        attributions_mean_fc = cal_mean_attribution(attributions_ixg_fc)
        peak_attr_fc, peak_flag_fc = get_peaks(attributions_mean_fc, chromosome, seq_start, seq_end, gene, pert)
        peaks = pd.concat([peaks, peak_attr])
        peaks_fc = pd.concat([peaks_fc, peak_attr_fc])
        for i,j in zip(range(0, len(attributions_ixg_fc), nbin), peak_flag_fc):
            if j == 1 or j == -1:
                split_score = attributions_ixg[i:i+nbin]
                split_score_fc = attributions_ixg_fc[i:i+nbin]
                attributions_score.append(split_score.detach().numpy())
                attributions_score_fc.append(split_score_fc.detach().numpy())
    if save == "pert":
        print(f"attribution_seq/{study}/{pert}{suffix}")
        os.makedirs(f"attribution_seq/{study}/{pert}{suffix}", exist_ok=True)
        peaks.to_csv(f"attribution_seq/{study}/{pert}{suffix}/00_all_{pert}_{attribution}_cpm.bed", index=False, header=False, sep="\t")
        peaks_fc.to_csv(f"attribution_seq/{study}/{pert}{suffix}/00_all_{pert}_{attribution}_fc.bed", index=False, header=False, sep="\t")
        with h5py.File(f"attribution_seq/{study}/{pert}{suffix}/{pert}{suffix}.h5", 'w') as hf:
            combined_array1 = np.array(attributions_score)
            hf.create_dataset(f"{attribution}_cpm", data=combined_array1)
            combined_array2 = np.array(attributions_score_fc)
            hf.create_dataset(f"{attribution}_fc", data=combined_array2)
    elif save == "seq":
        os.makedirs(f"attribution_seq/{study}/{gene}{suffix}", exist_ok=True)
        peaks.to_csv(f"attribution_seq/{study}/{gene}/00_all_{gene}_{attribution}_cpm.bed", index=False, header=False, sep="\t")
        peaks_fc.to_csv(f"attribution_seq/{study}/{gene}/00_all_{gene}_{attribution}_fc.bed", index=False, header=False, sep="\t")
        with h5py.File(f"attribution_seq/{study}/{gene}{suffix}/{gene}{suffix}.h5", 'w') as hf:
            combined_array1 = np.array(attributions_score)
            hf.create_dataset(f"{attribution}_cpm", data=combined_array1)
            combined_array2 = np.array(attributions_score_fc)
            hf.create_dataset(f"{attribution}_fc", data=combined_array2)



direction = sys.argv[4]
target_dat = sys.argv[5]

# attribution_pert ##
if direction == "pert":
    modisco = True
    suffix = ""
    cor = pd.read_csv(f'figures/{study}/cor_matrix/correlation_across_perts.txt', sep="\t")
    if target_dat == "all":
        inputs = df3.columns.to_list()
    elif target_dat == "tf":
        tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].to_list()
        inputs = df3.T[df3.columns.str.contains("|".join(tf_list)) == True].index.to_list()
    elif target_dat == "condition":
        modisco = False
        suffix = ".test"
        inputs = ['Norman.CEBPA', 'Norman.HNF4A', 'Norman.TP73', 'Norman.IRF1', 'Norman.AHR', 'Norman.SPI1', 'Norman.KMT2A', 'Norman.PRDM1', 'Norman.CEBPB', 'Norman.SNAI1', 'Norman.FOXA1', 'Norman.JUN', 'Norman.ETS2', 'Norman.EGR1']
        genes = cor.query('training == "test"')["Gene"].to_list()
    for pert in inputs:
        if target_dat in ["all", "tf"]:
            genes = df3.abs().sort_values(pert, ascending=False).loc[:,pert].head(100).index.to_list()
        perturbations = [pert] * len(genes)
        main_process(perturbations, genes, study, suffix, modisco, save="pert")
elif direction == "seq":
    modisco = False
    suffix = ""
    perturbations = df3.columns.to_list()
    cor = pd.read_csv(f'figures/{study}/cor_matrix/correlation_across_perts.txt', sep="\t")
    if target_dat == "all":
        inputs = cor.query('training == "test"')["Gene"].to_list()
    elif target_dat == "top":
        inputs = cor.query('training == "test"').head(10)["Gene"].to_list()
    elif target_dat == "condition":
        inputs = ['ECH1', 'HNRNPL', 'RINL', 'NFKBIB', 'SIRT2', 'CCER2', 'MRPS12', 'SARS2', 'FBXO17', 'ZNF579', 'ZNF524', 'FIZ1', 'ZNF865', 'ZNF784', 'ZNF580', 'ZNF581', 'CCDC106', 'U2AF2', 'EPN1']
    for gene in inputs:
        genes = [gene] * len(perturbations)
        main_process(perturbations, genes, study, suffix, modisco, save="seq")




