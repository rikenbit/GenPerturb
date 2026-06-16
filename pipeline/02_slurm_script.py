import pandas as pd
import numpy as np
from genperturb.model._genperturb import GenPerturb
from genperturb.evaluation._model_stats import ModelStats
#from genperturb.evaluation._model_stats_tpm2fc import ModelStatsFC
import subprocess
import sys
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("study_name", type=str)
parser.add_argument("plan", type=str)
parser.add_argument("pretrained_model", type=str)
parser.add_argument("--lora_r", type=int, default=256)
parser.add_argument("--lora_alpha", type=int, default=2)
args = parser.parse_args()

study_name = args.study_name
plan = args.plan
pretrained_model = args.pretrained_model
lora_r = args.lora_r
lora_alpha = args.lora_alpha
study_suffix_extra = os.environ.get("GENPERTURB_STUDY_SUFFIX_EXTRA", "")
epoch_override = os.environ.get("GENPERTURB_EPOCH_OVERRIDE", "")


def with_suffix_extra(study):
    if study_suffix_extra:
        return f"{study}{study_suffix_extra}"
    return study


def override_epoch(default_epoch):
    if not epoch_override:
        return default_epoch
    return int(epoch_override)

#study_name = "NormanWeissman2019_filtered_mixscape_exnp_train"
#plan = "transfer"
#pretrained_model = "enformer"
#pretrained_model = "alphagenome"


df    = pd.read_csv(f'data/{study_name}.tsv', sep="\t", index_col=[0])
bed   = f'fasta/{study_name}.bed'
fasta = f'fasta/GRCh38.p14.genome.fa'


if pretrained_model == "enformer":
    context_length = 196_608
    hdf5  = f'data/{study_name}_enformer.h5'
    emb_method = 'tss'
elif pretrained_model == "borzoi":
    context_length = 524_288
    hdf5  = f'data/{study_name}_borzoi.h5'
    emb_method = 'tss'
elif pretrained_model == "alphagenome":
    context_length = 1_048_576
    hdf5 = f'data/{study_name}_alphagenome.h5'
    emb_method = 'tss'
elif pretrained_model.startswith("alphagenome_fold_"):
    context_length = 1_048_576
    hdf5 = f'data/{study_name}_{pretrained_model}.h5'
    df = pd.read_csv(f'data/{study_name}_{pretrained_model}.tsv', sep="\t", index_col=[0])
    bed = f'fasta/{study_name}_{pretrained_model}.bed'
    emb_method = 'tss'
elif pretrained_model == "simplecnn":
    context_length = 40001
    hdf5 = None
    emb_method = 'tss'
    #df = pd.read_csv(f'data/{study_name}_nonperturb.tsv', sep="\t", index_col=[0])


def cal_model_stats(study, df, pred, pretrained_model, load_stats=False):
    modelstats = ModelStats(study, df, pred, pretrained_model)
    modelstats.main(load_stats=load_stats)

training = True
#training = False

if plan == "transfer":
    epoch = override_epoch(100)
    batch = 256
    study = with_suffix_extra(f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3')
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_nonperturb'
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_moe_onlydna_2'
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_moe_split_se16k2_1'
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_norm_quality_based_routing'
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_negcorloss_noshuffle'
    #study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3_masked'
    if training:
        model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained=pretrained_model, training_method=plan, study=study, emb_method=emb_method)
        model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1)
        model.load_model()
        pred = model.impute(hdf5=hdf5, context_length=context_length, batch_size=batch)
        os.makedirs(f"prediction/{study}/", exist_ok=True)
        np.save(f"prediction/{study}/prediction.npy", pred)
        cal_model_stats(study, df, pred, pretrained_model, load_stats=False)
    else:
        pred = np.load(f"prediction/{study}/prediction.npy")
        #cal_model_stats(study, df, pred, pretrained_model, load_stats=True)
        cal_model_stats(study, df, pred, pretrained_model, load_stats=False)

elif plan in ["finetuning", "lora"]:
    target_length = 4
    if training:
        if plan == "lora":
            epoch = override_epoch(100)
            batch = 2
            study = with_suffix_extra(f'{study_name}__{pretrained_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_lora_r{lora_r}a{lora_alpha}')
            model = GenPerturb(df, bed=bed, fasta=fasta, context_length=context_length,
                pretrained=pretrained_model, training_method=plan, target_length=target_length, study=study, emb_method=emb_method)
            model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1, accumulate=256//batch,
                        lora_r=lora_r, lora_alpha=lora_alpha)
        else:
            epoch = override_epoch(150)
            batch = 2
            #study = f'{study_name}__{pretrained_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3'
            study = with_suffix_extra(f'{study_name}__{pretrained_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_full')
            #study = f'{study_name}__{pretrained_model}_finetuning_epoch{epoch}_batch{batch}_adamw5e3_plr1e10'
            model = GenPerturb(df, bed=bed, fasta=fasta, context_length=context_length,
                pretrained=pretrained_model, training_method=plan, target_length=target_length, study=study, emb_method=emb_method)
            model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1, accumulate=256//batch)
            model.load_model()
        pred = model.impute(bed=bed, fasta=fasta, context_length=context_length, batch_size=batch*1)
        os.makedirs(f"prediction/{study}/", exist_ok=True)
        np.save(f"prediction/{study}/prediction.npy", pred)
        cal_model_stats(study, df, pred, pretrained_model, load_stats=False)
    else:
        epoch = 150
        batch = 32
        study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
        pred = np.load(f"prediction/{study}/prediction.npy")
        cal_model_stats(study, df, pred, pretrained_model, load_stats=True)

elif plan == "prediction":
    epoch = 100
    batch = 256
    impute_batch = 16
    study = f'{study_name}__{pretrained_model}_transfer_epoch{epoch}_batch{batch}_adamw5e3'
    model = GenPerturb(df, hdf5=hdf5, context_length=context_length, pretrained=pretrained_model, training_method="transfer", study=study, emb_method=emb_method)
    model.load_model()
    model.load_pretrained_model()
    model.training_method = plan
    model.module.training_method = plan
    pred = model.impute(bed=bed, fasta=fasta, context_length=context_length, batch_size=impute_batch)
    study_pred = f'{study}_fromdna'
    os.makedirs(f"prediction/{study_pred}/", exist_ok=True)
    np.save(f"prediction/{study_pred}/prediction.npy", pred)
    cal_model_stats(study_pred, df, pred, pretrained_model)

elif plan == "baseline":
    target_length = 4
    epoch = override_epoch(150)
    batch = 2
    study = with_suffix_extra(f'{study_name}__{pretrained_model}_baseline_epoch{epoch}_batch{batch}_adamw5e3')
    if training:
        model = GenPerturb(df, bed=bed, fasta=fasta, context_length=context_length,
            pretrained=pretrained_model, training_method=plan, target_length=target_length, study=study, emb_method=emb_method)
        model.train(max_epochs=epoch, batch_size=batch, use_device="gpu", gpus=1, accumulate=256//batch)
        model.load_model()
        pred = model.impute(bed=bed, fasta=fasta, context_length=context_length, batch_size=batch*1)
        os.makedirs(f"prediction/{study}/", exist_ok=True)
        np.save(f"prediction/{study}/prediction.npy", pred)
        cal_model_stats(study, df, pred, pretrained_model, load_stats=False)
    else:
        pred = np.load(f"prediction/{study}/prediction.npy")
        cal_model_stats(study, df, pred, pretrained_model, load_stats=False)

else:
    raise ValueError(f"Unsupported training plan: {plan}")
