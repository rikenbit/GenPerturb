import pandas as pd
import numpy as np
from genperturb.model._genperturb import GenPerturb
from genperturb.evaluation._model_stats import ModelStats
from genperturb.evaluation._model_stats_tpm2fc import ModelStatsFC
import subprocess
import sys
import os


study_name = sys.argv[1]
plan = sys.argv[2]
pretraind_model = sys.argv[3]

#study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
#plan = "finetuning"
#pretraind_model = "enformer"

df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
bed   = f'fasta/{study_name}.bed'
fasta = f'fasta/GRCh38.p13.genome.fa'

remove_ov_tss = False

if remove_ov_tss:
    df    = pd.read_csv(f'data/{study_name}_short_tss10kbprm.tsv', sep="\t", index_col=[0])
    bed   = f'fasta/{study_name}_short_tss10kbprm.bed'


if pretraind_model == "enformer":
    context_length = 196_608
    hdf5  = f'data/{study_name}.h5'
    #hdf5  = f'data/{study_name}_enformer_masked.h5'
    emb_method = 'tss'
elif pretraind_model == "hyena_dna_tss":
    context_length = 159_998
    hdf5 = f'data/{study_name}_hyena_tss.h5'
    emb_method = 'tss'
elif pretraind_model == "hyena_dna_last":
    context_length = 159_998
    hdf5 = f'data/{study_name}_hyena_last.h5'
    emb_method = 'last'
elif pretraind_model == "hyena_dna_mean":
    context_length = 159_998
    hdf5 = f'data/{study_name}_hyena_mean.h5'
    emb_method = 'mean'
elif pretraind_model == "nucleotide_transformer_tss":
    context_length = 12_282
    hdf5 = f'data/{study_name}_nt_tss.h5'
    emb_method = 'tss'
elif pretraind_model == "nucleotide_transformer_cls":
    context_length = 12_282
    hdf5 = f'data/{study_name}_nt_cls.h5'
    emb_method = 'cls'
elif pretraind_model == "nucleotide_transformer_mean":
    context_length = 12_282
    hdf5 = f'data/{study_name}_nt_mean.h5'
    emb_method = 'mean'



def cal_model_stats(study, df, pred, pretraind_model, load_stats=False):
    modelstats = ModelStats(study, df, pred, pretraind_model)
    modelstats.main(load_stats=load_stats)
    modelstatsfc = ModelStatsFC(study, df, pred, pretraind_model)
    modelstatsfc.main(load_stats=load_stats)

training = True
#training = False

if plan == "transfer":
    epoch = 100
    batch = 256
    study = f'{study_name}__{pretraind_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
    #study = f'{study_name}__{pretraind_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_masked'
    if training:
        model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained=pretraind_model, training_method=plan, study=study, emb_method=emb_method)
        model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1)
        model.load_model()
        pred = model.impute(hdf5=hdf5, context_length=context_length, batch_size=batch)
        os.makedirs(f"prediction/{study}/", exist_ok=True)
        np.save(f"prediction/{study}/prediction.npy", pred)
        cal_model_stats(study, df, pred, pretraind_model, load_stats=False)
    else:
        pred = np.load(f"prediction/{study}/prediction.npy")
        #cal_model_stats(study, df, pred, pretraind_model, load_stats=True)
        cal_model_stats(study, df, pred, pretraind_model, load_stats=False)

elif plan == "prediction":
    epoch = 100
    batch = 256
    impute_batch = 16
    study = f'{study_name}__{pretraind_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
    model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained=pretraind_model, training_method="transfer", study=study, emb_method=emb_method)
    model.load_model()
    model.load_pretrained_model()
    model.training_method = plan
    model.module.training_method = plan
    pred = model.impute(bed=bed, fasta=fasta, context_length=context_length, batch_size=impute_batch)
    study_pred = f'{study}_fromdna'
    os.makedirs(f"prediction/{study_pred}/", exist_ok=True)
    np.save(f"prediction/{study_pred}/prediction.npy", pred)
    cal_model_stats(study_pred, df, pred, pretraind_model)


elif plan in ["finetuning", "lora"]:
    target_length = 4
    if training:
        if plan == "lora":
            epoch = 20
            batch = 8
            study = f'{study_name}__{pretraind_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_lora_r256a2'
            model = GenPerturb(df, bed=bed, fasta=fasta, context_length=context_length,
                pretrained=pretraind_model, training_method=plan, target_length=target_length, study=study, emb_method=emb_method)
            model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1, accumulate=32)
            model.load_lora_model()
        else:
            epoch = 40
            batch = 2
            study = f'{study_name}__{pretraind_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_full'
            #study = f'{study_name}__{pretraind_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_plr1e10'
            model = GenPerturb(df, bed=bed, fasta=fasta, context_length=context_length,
                pretrained=pretraind_model, training_method=plan, target_length=target_length, study=study, emb_method=emb_method)
            model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1, accumulate=128)
            model.load_model()
        pred = model.impute(bed=bed, fasta=fasta, context_length=context_length, batch_size=batch*1)
        os.makedirs(f"prediction/{study}/", exist_ok=True)
        np.save(f"prediction/{study}/prediction.npy", pred)
        cal_model_stats(study, df, pred, pretraind_model, load_stats=False)
    else:
        pred = np.load(f"prediction/{study}/prediction.npy")
        cal_model_stats(study, df, pred, pretraind_model, load_stats=True)
