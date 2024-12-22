import pandas as pd
import numpy as np
import sys
import os


## study name ##
study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"

## select gene and pert ##
epoch = 100
batch = 256

cor_seqs_dict = {}
cor_perts_dict = {}
cor_seqs_fc_dict = {}
cor_perts_fc_dict = {}
preds = {}

pretrained_models = [
"enformer",
"hyena_dna_tss",
"hyena_dna_last",
#"hyena_dna_mean",
"nucleotide_transformer_tss",
"nucleotide_transformer_cls",
#"nucleotide_transformer_mean",
]



for pretrained_model in pretrained_models:
    study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
    cor_seqs_dict[pretrained_model] = pd.DataFrame()
    cor_perts_dict[pretrained_model] = pd.DataFrame()
    cor_seqs_fc_dict[pretrained_model] = pd.DataFrame()
    cor_perts_fc_dict[pretrained_model] = pd.DataFrame()
    cor_seq = pd.read_csv(f"figures/{study}/cor_matrix/correlation.txt", sep="\t")
    cor_pert = pd.read_csv(f"figures/{study}/cor_matrix/correlation_pert.txt", sep="\t")
    cor_seq_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation.txt", sep="\t")
    cor_pert_fc = pd.read_csv(f"figures/{study}/cor_matrix_fc/correlation_pert.txt", sep="\t")
    cor_seqs_dict[pretrained_model] = cor_seq
    cor_perts_dict[pretrained_model] = cor_pert
    cor_seqs_fc_dict[pretrained_model] = cor_seq_fc
    cor_perts_fc_dict[pretrained_model] = cor_pert_fc
    df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
    pred = np.load(f"prediction/{study}/prediction.npy")
    df2 = pd.DataFrame(pred)
    df2.columns = df.columns[1:]
    df2.index = df.index
    ctrl = df2.columns[0]
    preds[pretrained_model] = (df2.T - df2[ctrl]).T.drop(ctrl, axis=1).copy()


# select gene
cor_list = [cor_seqs_fc_dict[i].set_index(["Gene", "training"]).loc[:, ["Correlation"]].rename(columns={"Correlation":f"{i}"}) for i in pretrained_models]
cor = pd.concat([df[~df.index.duplicated()] for df in cor_list], axis=1)
cor["mean"] = cor.mean(1)
genes = [i[0] for i in cor.query('training == "test"').sort_values("mean", ascending=False).head(10).index]

>>> cor.query('training == "test"').sort_values("mean", ascending=False).head(10)
                  enformer  hyena_dna_tss  hyena_dna_last  nucleotide_transformer_tss  nucleotide_transformer_cls      mean
Gene    training                                                                                                           
TYROBP  test      0.924786       0.370675        0.513475                    0.475754                    0.567853  0.570509
GIPR    test      0.668249       0.428668        0.619154                    0.424984                    0.566533  0.541518
ACAP2   test      0.830926       0.305732        0.534371                    0.311852                    0.598223  0.516221
PHF21A  test      0.641510       0.485274        0.506477                    0.446305                    0.464544  0.508822
CCDC106 test      0.678484       0.398695        0.566549                    0.238429                    0.623036  0.501039
HIPK3   test      0.697814       0.461401        0.522704                    0.186127                    0.603871  0.494383
BCL3    test      0.676173       0.350133        0.516858                    0.372265                    0.513381  0.485762
PPM1N   test      0.567556       0.478305        0.563945                    0.307843                    0.455947  0.474719
RASGRP4 test      0.624634       0.359998        0.524767                    0.379531                    0.477914  0.473369
RBM23   test      0.670804       0.429107        0.543289                    0.180721                    0.539915  0.472767


# select pert
genes = cor.query('training == "test"').sort_values("mean", ascending=False).head(10).reset_index()["Gene"].to_list()
for gene in genes:
    print("--------------")
    print(gene)
    pred = pd.concat([preds[i].T.loc[:,[gene]].rename(columns={gene:i}) for i in pretrained_models], axis=1)
    pred["mean"] = pred.mean(axis=1)
    pred = pred.sort_values("mean", ascending=False)
    pred.sort_values("mean", ascending=False).head(5)


"TYROBP"
"ACAP2"
"HIPK3"
"BCL3"
"RASGRP4"





