import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import pandas as pd
import numpy as np
from enformer_pytorch import Enformer, GenomeIntervalDataset, seq_indices_to_one_hot
import pyBigWig
from pybedtools import BedTool
import h5py
from scipy.stats import zscore
import os
import sys
from captum.attr import Saliency, DeepLift, IntegratedGradients, InputXGradient
from genperturb.model._genperturb import GenPerturb


study_name = sys.argv[1]
#study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
hdf5  = f'data/{study_name}.h5'
bed   = f'fasta/{study_name}.bed'
fasta = f'fasta/GRCh38.p13.genome.fa'

context_length = 196_608
emb_method = 'tss'

study_suffix = sys.argv[2]
#study_suffix = "enformer_transfer_epoch100_batch256_adamw5e3"
study = f'{study_name}__{study_suffix}'
model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained="enformer", training_method="transfer", study=study)
model.load_model()
model.load_pretrained_model()
model = model.cuda()

ds = GenomeIntervalDataset(bed_file=bed, fasta_file=fasta, return_seq_indices=True, context_length=context_length)

pred = np.load(f"prediction/{study}/prediction.npy")
df2 = pd.DataFrame(pred)
df2.columns = df.columns[1:]
df2.index = df.index

ctrl = df2.columns[0]
df3 = (df2.T - df2[ctrl]).T.drop(ctrl, axis=1).copy()


def cal_attribution(model, ds, pert_num, gene_num, attribution="ixg", context_length=196608):
    input_seq = seq_indices_to_one_hot(ds[gene_num]).float().requires_grad_(True).cuda()
    baseline  = seq_indices_to_one_hot(torch.randint(0, 4, size=[1,context_length])[0]).float().requires_grad_(True).cuda()
    if attribution == "ixg":
        print("InputXGradient")
        ixg = InputXGradient(model)
        attributions_ixg = ixg.attribute(input_seq, target=pert_num)
        return attributions_ixg.cpu()
    elif attribution == "ig":
        print("IntegratedGradients")
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(input_seq, baseline, target=pert_num, return_convergence_delta=True, internal_batch_size=1000, n_steps=400)
        return attributions.cpu(), delta.cpu()
    elif attribution == "sa":
        print("Saliency")
        sa = Saliency(model)
        attributions = sa.attribute(input_seq, target=pert_num, abs=False)
        return attributions.cpu()
    elif attribution == "dl":
        dl = DeepLift(model)
        attributions, delta = dl.attribute(input_seq, baseline, target=pert_num, return_convergence_delta=True)
        return attributions.cpu(), delta.cpu()

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
    ctrl_attributions_ixg = cal_attribution(model, ds, pert_num, gene_num, attribution=attribution, context_length=context_length)
    ctrl_attributions_sa  = cal_attribution(model, ds, pert_num, gene_num, attribution="sa", context_length=context_length)
    return ctrl_attributions_ixg, ctrl_attributions_sa

def main_process(perturbations, genes, study, suffix="", modisco=False, save="", attribution="ixg", nbin=128):
    attributions_score = []
    attributions_score_fc = []
    attributions_hyp = []
    peaks    = pd.DataFrame()
    peaks_fc = pd.DataFrame()
    for pert, gene in zip(perturbations, genes):
        print(f"{pert} {gene}")
        if modisco:
            ctrl_attributions_ixg, ctrl_attributions_sa = ctrl_process(gene)
        else:
            if "ctrl_attributions_ixg" not in locals():
                ctrl_attributions_ixg, ctrl_attributions_sa = ctrl_process(gene)
        pert_num = df2.columns.get_loc(pert)
        gene_num = ds.df.with_row_index("row_number").filter(pl.col("column_4") == gene)["row_number"][0]
        chromosome = ds.df[gene_num, 0]
        seq_start  = int(ds.df[gene_num, 1] - (context_length / 2))
        seq_end    = int(ds.df[gene_num, 1] + (context_length / 2))
        attributions_ixg = cal_attribution(model, ds, pert_num, gene_num, attribution=attribution, context_length=context_length)
        attributions_mean = cal_mean_attribution(attributions_ixg.sum(1))
        peak_attr, peak_flag = get_peaks(attributions_mean, chromosome, seq_start, seq_end, gene, pert)
        attributions_ixg_fc = attributions_ixg - ctrl_attributions_ixg
        attributions_mean_fc = cal_mean_attribution(attributions_ixg_fc.sum(1))
        peak_attr_fc, peak_flag_fc = get_peaks(attributions_mean_fc, chromosome, seq_start, seq_end, gene, pert)
        attributions_sa = cal_attribution(model, ds, pert_num, gene_num, attribution="sa", context_length=context_length)
        attributions_sa = attributions_sa - ctrl_attributions_sa
        attributions_sa = attributions_sa - attributions_sa.mean(1, keepdims=True) # normalize mean to zero
        if modisco:
            peaks_fc = pd.concat([peaks_fc, peak_attr_fc.query('peak == 1 | peak == -1')])
        else:
            peaks = pd.concat([peaks, peak_attr])
            peaks_fc = pd.concat([peaks_fc, peak_attr_fc])
        for i,j in zip(range(0, len(attributions_ixg_fc), nbin), peak_flag_fc):
            if j == 1 or j == -1:
                split_score = attributions_ixg[i:i+nbin]
                split_score_fc = attributions_ixg_fc[i:i+nbin]
                split_hyp = attributions_sa[i:i+nbin]
                attributions_score.append(split_score.detach().numpy())
                attributions_score_fc.append(split_score_fc.detach().numpy())
                attributions_hyp.append(split_hyp.detach().numpy())
    if save == "pert_modisco":
        os.makedirs(f'attribution_pert/{study}/{pert.replace("/", "_")}', exist_ok=True)
        peaks_fc.to_csv(f'attribution_pert/{study}/{pert.replace("/", "_")}/00_all_{pert.replace("/", "_")}_{attribution}_fc.bed', index=False, header=False, sep="\t")
        os.makedirs(f'tfmodisco/{study}/{pert.replace("/", "_")}{suffix}', exist_ok=True)
        with h5py.File(f'tfmodisco/{study}/{pert.replace("/", "_")}{suffix}/{pert.replace("/", "_")}{suffix}.h5', 'w') as hf:
            combined_array1 = np.array(attributions_score_fc)
            hf.create_dataset(attribution, data=combined_array1)
            combined_array2 = np.array(attributions_hyp)
            hf.create_dataset('saliency', data=combined_array2)
    elif save == "seq":
        os.makedirs(f"attribution_seq/{study}/{gene}{suffix}", exist_ok=True)
        peaks.to_csv(f"attribution_seq/{study}/{gene}/00_all_{gene}_{attribution}_cpm.bed", index=False, header=False, sep="\t")
        peaks_fc.to_csv(f"attribution_seq/{study}/{gene}/00_all_{gene}_{attribution}_fc.bed", index=False, header=False, sep="\t")
        with h5py.File(f"attribution_seq/{study}/{gene}{suffix}/{gene}{suffix}.h5", 'w') as hf:
            combined_array1 = np.array(attributions_score)
            hf.create_dataset(f"{attribution}_cpm", data=combined_array1)
            combined_array2 = np.array(attributions_score_fc)
            hf.create_dataset(f"{attribution}_fc", data=combined_array2)
    elif save == "pert":
        print(f"attribution_seq/{study}/{pert}{suffix}")
        os.makedirs(f"attribution_seq/{study}/{pert}{suffix}", exist_ok=True)
        peaks.to_csv(f"attribution_seq/{study}/{pert}{suffix}/00_all_{pert}_{attribution}_cpm.bed", index=False, header=False, sep="\t")
        peaks_fc.to_csv(f"attribution_seq/{study}/{pert}{suffix}/00_all_{pert}_{attribution}_fc.bed", index=False, header=False, sep="\t")
        with h5py.File(f"attribution_seq/{study}/{pert}{suffix}/{pert}{suffix}.h5", 'w') as hf:
            combined_array1 = np.array(attributions_score)
            hf.create_dataset(f"{attribution}_cpm", data=combined_array1)
            combined_array2 = np.array(attributions_score_fc)
            hf.create_dataset(f"{attribution}_fc", data=combined_array2)

direction = sys.argv[3]
target_dat = sys.argv[4]

# attribution_pert ##
if direction == "pert":
    modisco = True
    suffix = ""
    if target_dat == "all":
        inputs = df3.columns.to_list()
    elif target_dat == "tf":
        tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].to_list()
        inputs = df3.T[df3.columns.str.contains("|".join(tf_list)) == True].index.to_list()
    elif target_dat == "tf_allgene":
        tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].to_list()
        inputs = df3.T[df3.columns.str.contains("|".join(tf_list)) == True].index.to_list()
    elif target_dat == "condition":
        inputs = ["Norman.CEBPA"]
    for pert in inputs:
        genes = df3.abs().sort_values(pert, ascending=False).loc[:,pert].head(150).index.to_list()
        perturbations = [pert] * len(genes)
        main_process(perturbations, genes, study, suffix, modisco, save="pert_modisco")
elif direction == "seq":
    modisco = False
    suffix = ""
    perturbations = df3.columns.to_list()
    cor = pd.read_csv(f'figures/{study}/cor_matrix/correlation_across_perts.txt', sep="\t")
    if target_dat == "all":
        inputs = cor.query('training == "test"')["Gene"].to_list()
        #inputs = cor.query('training == "val"')["Gene"].to_list()
        #inputs = cor.query('training == "train"')["Gene"].to_list()
    elif target_dat == "top":
        inputs = cor.query('training == "test"').head(10)["Gene"].to_list()
    elif target_dat == "condition":
        inputs = ['ECH1', 'HNRNPL', 'RINL', 'NFKBIB', 'SIRT2', 'CCER2', 'MRPS12', 'SARS2', 'FBXO17', 'ZNF579', 'ZNF524', 'FIZ1', 'ZNF865', 'ZNF784', 'ZNF580', 'ZNF581', 'CCDC106', 'U2AF2', 'EPN1']
    for gene in inputs:
        genes = [gene] * len(perturbations)
        main_process(perturbations, genes, study, suffix, modisco, save="seq")
elif direction == "pert_all":
    modisco = False
    suffix = ".test"
    #suffix = ".all"
    cor = pd.read_csv(f'figures/{study}/cor_matrix/correlation_across_perts.txt', sep="\t")
    if target_dat == "tf_allgene":
        tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"])["HGNC symbol"].to_list()
        inputs = df3.T[df3.columns.str.contains("|".join(tf_list)) == True].index.to_list()
    elif target_dat == "test":
        input_perts = ["Norman.HNF4A", "Norman.IRF1", "Norman.CEBPA", "Norman.TP73", "Norman.FOXA1", "Norman.AHR", "Norman.PRDM1", "Norman.SPI1", "Norman.SNAI1", "Norman.KMT2A", "Norman.CEBPB", 'Norman.JUN', 'Norman.ETS2', 'Norman.EGR1']
        input_genes  = cor.query('training == "test"')["Gene"].to_list()
    for pert in input_perts:
        genes = input_genes
        perturbations = [pert] * len(genes)
        main_process(perturbations, genes, study, suffix, modisco, save="pert")
    



