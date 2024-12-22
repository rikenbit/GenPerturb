import logging
from math import ceil
from typing import Dict, List, Optional, Union
import os
import glob

import pandas as pd
import numpy as np
import torch
from torch import nn

from genperturb.module._genperturb_torch import GenPerturbTorch
from genperturb.dataloaders._joint_dataloader import JointDataLoader, JointDataLoaderPred
from genperturb.train._train import Training, LoadTrainedModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.strategies import DeepSpeedStrategy
from peft import LoraConfig, get_peft_model, PeftModel

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class GenPerturb(nn.Module):
    def __init__(
        self,
        df : pd.DataFrame,
        hdf5 : Optional[str] = None,
        bed : Optional[str] = None,
        fasta : Optional[str] = None,
        context_length : int = 64_128,
        pretrained : Optional[str] = "enformer",
        emb_method : Optional[str] = "target",
        training_method : Optional[str] = "transfer",
        target_length : int = 4,
        study : str = "tmp"
    ):

        super().__init__()

        self.study = study

        self.df = df
        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.context_length = context_length
        self.pretrained = pretrained
        self.emb_method = emb_method
        self.training_method = training_method
        self.target_length = target_length

        self.load_lora = False

        self.num_perturb = len(df.columns) - 1


        if self.pretrained == "enformer":
            self.checkpoint = 'EleutherAI/enformer-official-rough'
            self.return_sequence=False
        elif self.pretrained in ["hyena_dna_tss", "hyena_dna_last"]:
            self.checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
            self.return_sequence=True
        elif self.pretrained in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
            self.checkpoint = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
            self.return_sequence=True

        if self.training_method in ["finetuning", "lora"]:
            self.module = GenPerturbTorch(num_perturb=self.num_perturb,
                                        pretrained=self.pretrained,
                                        emb_method=self.emb_method,
                                        target_length=self.target_length,
                                        training_method=self.training_method)

            self.load_pretrained_model()
            self.load_tokenizer()
            print("finetuning : done")
            print("Using bed : ", bed)
            print("Using fasta : ", fasta)
        elif self.training_method == "transfer":
            self.module = GenPerturbTorch(num_perturb=self.num_perturb,
                                        pretrained=self.pretrained,
                                        emb_method=self.emb_method,
                                        target_length=self.target_length,
                                        training_method=self.training_method)

            self.load_tokenizer()
            print("full training and finetuning : not done")
            print("Using hdf5 : ", hdf5)

    def train(
        self, 
        max_epochs: int = 10,
        batch_size : int = 1,
        use_device: str = "cpu",
        gpus: int = 1,
        nnodes: int = 1,
        finetuning: str = None,
        strategy: str = "auto",
        accumulate: int = 1,
        save_dir: str = "./logs/"
    ):

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.strategy   = strategy

        print(f"fine turning method : {finetuning}")
        print(f"strategy : {strategy}")

        os.makedirs(save_dir, exist_ok=True)
        csv_logger = CSVLogger(save_dir=save_dir, name=self.study)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=save_dir + self.study,
            filename='model-{epoch:02d}-{val:.2f}',
            save_top_k=1,
            save_last=True,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
        )

        

        if self.training_method == "transfer":
            dataloader = JointDataLoader(
                self.df,
                hdf5=self.hdf5,
                context_length=self.context_length,
                batch_size=self.batch_size
            )
        elif self.training_method in ["finetuning", "lora"]:
            dataloader = JointDataLoader(
                self.df,
                bed=self.bed,
                fasta=self.fasta,
                return_sequence=self.return_sequence,
                context_length=self.context_length,
                batch_size=self.batch_size,
                tokenizer = self.module.tokenizer,
            )

        if self.training_method == "finetuning":
            module = Training(self.module, use_device=use_device, strategy=strategy, lr=1e-4)
        elif self.training_method == "lora":
            targe_modules  = ["to_q", "to_v"]
            config = LoraConfig(
                r=256,
                lora_alpha=2,
                bias="none",
                target_modules=targe_modules,
                modules_to_save=["_heads"],
            )

            self.lora_model = get_peft_model(self.module, config)
            module = Training(self.lora_model, use_device=use_device, strategy=strategy, lr=1e-4)
        elif self.training_method == "transfer":
            module = Training(self.module, use_device=use_device, lr=1e-4)


        if use_device == "cpu":
            self.trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accumulate_grad_batches=1,
                #accumulate_grad_batches=64,
                gradient_clip_val=0.2,
                callbacks=[checkpoint_callback, early_stop_callback],
                logger=[csv_logger],
            )
        elif use_device == "gpu":
            if self.training_method in ["finetuning", "lora"]:
                self.trainer = pl.Trainer(
                    max_epochs=self.max_epochs,
                    accelerator='gpu',
                    devices=gpus,
                    num_nodes=nnodes,
                    #strategy=strategy,
                    accumulate_grad_batches=accumulate,
                    precision=32,
                    gradient_clip_val=0.2,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    logger=[csv_logger],
                    limit_train_batches=50,
                    limit_val_batches=50,
                )
            elif self.training_method == "transfer":
                self.trainer = pl.Trainer(
                    max_epochs=self.max_epochs,
                    accelerator='gpu',
                    devices=gpus,
                    num_nodes=nnodes,
                    accumulate_grad_batches=accumulate,
                    precision=32,
                    gradient_clip_val=0.2,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    logger=[csv_logger],
                )
        else:
            print("No device")

        self.trainer.fit(module, dataloader)

        if self.training_method in ["lora"]:
            self.lora_model.save_pretrained(f"logs/{self.study}/lora_checkpoint")
            self.load_lora = True

        self.loss_plot()

    def loss_plot(self):
        version_dirs = [d for d in os.listdir(f"logs/{self.study}") if os.path.isdir(os.path.join(f"logs/{self.study}", d))]
        max_version = max([int(d.split("_")[1]) for d in version_dirs if d.startswith("version_")], default=0)
        file_path = f"logs/{self.study}/version_{max_version}/metrics.csv"
        data = pd.read_csv(file_path)
        train_loss = data.loc[:,["epoch", "train_loss_epoch"]].dropna()
        val_loss = data.loc[:,["epoch", "val_loss"]].dropna()
        
        os.makedirs(f"figures/{self.study}/loss", exist_ok=True)

        plt.rcParams["font.size"] = 18
        plt.plot("epoch", "train_loss_epoch", data=train_loss, label='Train Loss')
        plt.plot("epoch", "val_loss", data=val_loss, label='Validation Loss')
        plt.title('Train and Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/{self.study}/loss/loss_plot.png')
        plt.clf()

    def load_model(self, ckpt=None, deepspeed=False):
        if ckpt is None:
            file_list = glob.glob(f'logs/{self.study}/model*.ckpt')
            ckpt = max(file_list, key=os.path.getctime)

        if deepspeed:
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            self.module = load_state_dict_from_zero_checkpoint(self.module, ckpt)
        else:
            self.module = LoadTrainedModule.load_from_checkpoint(ckpt, module=self.module, map_location=torch.device('cpu')).module

    def load_lora_model(self):
        peft_checkpoint = f"logs/{self.study}/lora_checkpoint"
        self.lora_model = PeftModel.from_pretrained(self.module, peft_checkpoint)
        self.load_lora = True
    
    def load_pretrained_model(self):
        if self.pretrained == "enformer":
            from enformer_pytorch import Enformer
            self.module.pretrained_model = Enformer.from_pretrained(self.checkpoint, target_length=self.target_length)
            del self.module.pretrained_model._heads
        elif self.pretrained in ["hyena_dna_tss", "hyena_dna_last"]:
            from transformers import AutoModelForSequenceClassification
            self.module.pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, trust_remote_code=True)
        elif self.pretrained in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
            from transformers import AutoModelForMaskedLM
            self.module.pretrained_model = AutoModelForMaskedLM.from_pretrained(self.checkpoint, trust_remote_code=True)

    def load_tokenizer(self):
        if self.pretrained == "enformer":
            self.module.tokenizer = None
        elif self.pretrained in ["hyena_dna_tss", "hyena_dna_last"]:
            from transformers import AutoTokenizer
            self.module.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        elif self.pretrained in ["nucleotide_transformer_tss", "nucleotide_transformer_cls"]:
            from transformers import AutoTokenizer
            self.module.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)


    def impute(
        self,
        hdf5 : Optional[str] = None,
        bed : Optional[str] = None,
        fasta : Optional[str] = None,
        context_length : int = 196_608,
        mask = None,
        batch_size : int = 1,
        use_device: str = "gpu",
    ):

        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.context_length = context_length

        with torch.no_grad():
            x_pred = []

            if self.training_method in ["finetuning", "lora", "prediction"]:
                self.module.cuda()
                self.module.eval()

                dataloader = JointDataLoaderPred(
                    bed=self.bed,
                    fasta=self.fasta,
                    return_sequence=self.return_sequence,
                    context_length=self.context_length,
                    batch_size=batch_size,
                    mask=mask,
                    tokenizer = self.module.tokenizer,
                )

                for tensors in dataloader.dataloader():
                    out = self.module.forward(tensors[0].cuda(), cal_loss=False, prediction=True)
                    x_pred += [out.cpu()]

            elif self.training_method == "transfer":
                self.module.cuda()
                self.module.eval()
                if self.hdf5 is not None:
                    dataloader = JointDataLoaderPred(hdf5=hdf5, batch_size=batch_size)
                    for tensors in dataloader.dataloader():
                        out = self.module.forward(tensors[0].cuda(), cal_loss=False, prediction=True)
                        x_pred += [out.cpu()]

            x_pred_all = torch.cat(x_pred).squeeze().numpy()
            os.makedirs(f"prediction/{self.study}/", exist_ok=True)
            np.save(f"prediction/{self.study}/prediction.npy", x_pred_all)

            return x_pred_all

    def forward(self, x):
        self.module.eval()
        self.module.pretrained_model.eval()

        self.training_method = "prediction"
        self.module.training_method = "prediction"

        out = self.module.forward(x.cuda(), cal_loss=False, prediction=True)
        return out


