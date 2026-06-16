import logging
from math import ceil
from typing import Dict, List, Optional, Union
import os
import glob
import json

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from genperturb.module._genperturb_torch import GenPerturbTorch
from genperturb.dataloaders._joint_dataloader import JointDataLoader, JointDataLoaderPred
from genperturb.train._train import Training, LoadTrainedModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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


        if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
            self.checkpoint = (
                "all_folds"
                if self.pretrained == "alphagenome"
                else self.pretrained.removeprefix("alphagenome_")
            )
            self.return_sequence=False
        elif self.pretrained == "borzoi":
            self.checkpoint = 'johahi/borzoi-replicate-0'
            self.return_sequence=False
        elif self.pretrained == "enformer":
            self.checkpoint = 'EleutherAI/enformer-official-rough'
            self.return_sequence=False
        elif self.pretrained == "simplecnn":
            self.checkpoint = None
            self.return_sequence=False

        if self.training_method in ["finetuning", "lora", "baseline"]:
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

            #self.load_tokenizer()
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
        save_dir: str = "./logs/",
        lora_r: int = 256,
        lora_alpha: int = 2,
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
            dirpath=os.path.join(save_dir, self.study),
            filename='model-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
        )

        

        if self.training_method == "transfer":
            dataloader = JointDataLoader(
                self.df,
                hdf5=self.hdf5,
                context_length=self.context_length,
                batch_size=self.batch_size
            )
        elif self.training_method in ["finetuning", "lora", "baseline"]:
            dataloader = JointDataLoader(
                self.df,
                bed=self.bed,
                fasta=self.fasta,
                return_sequence=self.return_sequence,
                context_length=self.context_length,
                batch_size=self.batch_size,
                tokenizer = self.module.tokenizer,
                pretrained=self.pretrained,
            )

        if self.training_method in ["finetuning", "baseline"]:
            module = Training(self.module, use_device=use_device, strategy=strategy, lr=1e-4)
        elif self.training_method == "lora":
            self.configure_lora_model(lora_r=lora_r, lora_alpha=lora_alpha)
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
            if self.training_method in ["finetuning", "lora", "baseline"]:
                self.trainer = pl.Trainer(
                    max_epochs=self.max_epochs,
                    accelerator='gpu',
                    devices=gpus,
                    num_nodes=nnodes,
                    #strategy=strategy,
                    accumulate_grad_batches=accumulate,
                    precision="bf16-mixed",
                    gradient_clip_val=0.2,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    logger=[csv_logger],
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
            best_ckpt = checkpoint_callback.best_model_path
            if not best_ckpt:
                raise RuntimeError("Lightning did not report a best LoRA checkpoint.")

            self.save_lora_adapter_from_checkpoint(
                checkpoint_path=best_ckpt,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                output_dir=os.path.join(save_dir, self.study, "lora_checkpoint_best"),
                training_module=module,
                best_model_score=checkpoint_callback.best_model_score,
            )

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

    def configure_lora_model(self, lora_r=256, lora_alpha=2):
        from peft import LoraConfig, get_peft_model

        if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
            target_modules = ["to_qkv", "to_out"]
        else:
            target_modules = ["to_q", "to_v"]

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["_heads"],
        )
        self.lora_model = get_peft_model(self.module, config)
        return self.lora_model

    def save_lora_adapter_from_checkpoint(
        self,
        checkpoint_path,
        lora_r,
        lora_alpha,
        output_dir=None,
        training_module=None,
        best_model_score=None,
    ):
        if not hasattr(self, "lora_model"):
            self.configure_lora_model(lora_r=lora_r, lora_alpha=lora_alpha)

        if training_module is None:
            training_module = Training(self.lora_model, use_device="cpu", lr=1e-4)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        training_module.load_state_dict(checkpoint["state_dict"])

        if output_dir is None:
            output_dir = f"logs/{self.study}/lora_checkpoint_best"
        self.lora_model.save_pretrained(output_dir)

        if best_model_score is not None and hasattr(best_model_score, "item"):
            best_model_score = best_model_score.item()
        metadata = {
            "source_checkpoint": checkpoint_path,
            "best_model_score": best_model_score,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        }
        with open(os.path.join(output_dir, "best_checkpoint.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self.load_lora = True
        print(f"Saved LoRA adapter from Lightning checkpoint: {output_dir}")
        return output_dir

    def load_lora_model(self, peft_checkpoint=None):
        from peft import PeftModel

        if peft_checkpoint is None:
            candidates = [
                f"logs/{self.study}/lora_checkpoint_best",
                f"logs/{self.study}/lora_checkpoint",
            ]
            peft_checkpoint = next((path for path in candidates if os.path.isdir(path)), None)
            if peft_checkpoint is None:
                raise FileNotFoundError(
                    f"No LoRA adapter checkpoint found for {self.study}. "
                    f"Checked: {', '.join(candidates)}"
                )

        print(f"Loading LoRA adapter: {peft_checkpoint}")
        self.lora_model = PeftModel.from_pretrained(self.module, peft_checkpoint)
        self.load_lora = True
    
    def _enable_transformer_gradient_checkpointing(self, base_model):
        import types
        from torch.utils.checkpoint import checkpoint as ckpt

        tower = base_model.transformer_unet.transformer

        def checkpointed_forward(self, single):
            seq_len = single.shape[1]
            pairwise = None

            rel_pos_feats = self.rel_pos_features(single)

            if self.polar_pos_emb:
                polar_emb = self.polar_emb(seq_len)
                pos_emb = dict(polar_emb=polar_emb)
            else:
                rotary_emb = self.rotary_emb(seq_len)
                pos_emb = dict(rotary_emb=rotary_emb)

            for (attn, ff, maybe_s2p, maybe_pa, maybe_pf) in self.layers:
                def layer_fn(s, p,
                             _attn=attn, _ff=ff,
                             _s2p=maybe_s2p, _pa=maybe_pa, _pf=maybe_pf):
                    if _s2p is not None:
                        pw = _s2p(s, rel_pos_feats)
                        if p is not None:
                            pw = pw + p
                        p = _pa(pw) + pw
                        p = _pf(p) + p
                    s = _attn(s, pairwise=p, **pos_emb) + s
                    s = _ff(s) + s
                    return s, p

                single, pairwise = ckpt(layer_fn, single, pairwise, use_reentrant=False)

            return single, pairwise

        tower.forward = types.MethodType(checkpointed_forward, tower)

    def _enable_unet_gradient_checkpointing(self, base_model):
        from torch.utils.checkpoint import checkpoint as ckpt

        unet = base_model.transformer_unet

        orig_dna_fwd = unet.dna_embed.forward

        def ckpt_dna_embed(seq, _orig=orig_dna_fwd):
            return ckpt(_orig, seq, use_reentrant=False)
        unet.dna_embed.forward = ckpt_dna_embed

        for down in unet.downs:
            orig = down.forward

            def make_ckpt_down(original):
                def ckpt_forward(x, return_pre_pool=False):
                    def fn(x_in):
                        return original(x_in, return_pre_pool=return_pre_pool)
                    return ckpt(fn, x, use_reentrant=False)
                return ckpt_forward

            down.forward = make_ckpt_down(orig)

        for up in unet.ups:
            orig = up.forward

            def make_ckpt_up(original):
                def ckpt_forward(x, skip=None):
                    def fn(x_in, s):
                        return original(x_in, skip=s)
                    return ckpt(fn, x, skip, use_reentrant=False)
                return ckpt_forward

            up.forward = make_ckpt_up(orig)

    def load_pretrained_model(self):
        if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
            import os
            os.environ["ALPHAGENOME_TORCH_BF16"] = "1"
            from alphagenome_pytorch import AlphaGenome
            from alphagenome_pytorch.alphagenome import BatchRMSNorm
            import torch
            import torch.nn as nn
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            base_model = AlphaGenome()
            base_model.add_reference_heads("human")
            base_model.load_from_official_jax_model(self.checkpoint)
            base_model.to(device)

            for m in base_model.modules():
                if isinstance(m, BatchRMSNorm):
                    m.update_running_var = False

            self._enable_transformer_gradient_checkpointing(base_model)

            class AlphaGenomeEmbedder(nn.Module):
                def __init__(self, alphagenome_model):
                    super().__init__()
                    self.model = alphagenome_model
                    self.center_start = 4094
                    self.center_end = 4098
                    self._patch_dna_embed_for_onehot()

                def _patch_dna_embed_for_onehot(self):
                    from einops import rearrange
                    dna_embed = self.model.transformer_unet.dna_embed

                    def patched_forward(seq):
                        if seq.dim() == 3:
                            x = rearrange(seq.float(), 'b n d -> b d n')
                        else:
                            onehot = F.one_hot(seq, num_classes=dna_embed.dim_input).float()
                            x = rearrange(onehot, 'b n d -> b d n')
                        out = dna_embed.conv(x)
                        out = out + dna_embed.pointwise(out)
                        pooled = dna_embed.pool(out)
                        return pooled, out

                    dna_embed.forward = patched_forward

                def forward(self, x):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, embeddings_128bp, _ = self.model(
                            x,
                            organism_index=0,  # human
                            return_embeds=True
                        )
                        center_emb = embeddings_128bp[:, self.center_start:self.center_end, :]
                    return center_emb.float()

            embedder = AlphaGenomeEmbedder(base_model)
            self._enable_unet_gradient_checkpointing(base_model)
            self.module.pretrained_model = embedder

        elif self.pretrained == "borzoi":
            from huggingface_hub import hf_hub_download
            from borzoi_pytorch import Borzoi
            import torch, torch.nn as nn
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_model = Borzoi.from_pretrained(self.checkpoint, return_center_bins_only=True, bins_to_return=16)
            state_path = hf_hub_download(repo_id=self.checkpoint, filename="pytorch_model.bin")
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            base_model.to_empty(device=device)
            base_model.load_state_dict(state_dict, strict=True)

            class BorzoiEmbedder(nn.Module):
                def __init__(self, borzoi_model: Borzoi):
                    super().__init__()
                    self.borzoi = borzoi_model
                def forward(self, x):
                    z_crop  = self.borzoi.get_embs_after_crop(x)
                    z_final = self.borzoi.final_joined_convs(z_crop)
                    return z_final

            self.module.pretrained_model = BorzoiEmbedder(base_model)

        elif self.pretrained == "enformer":
            from enformer_pytorch import Enformer
            self.module.pretrained_model = Enformer.from_pretrained(self.checkpoint, target_length=self.target_length)
            del self.module.pretrained_model._heads

        elif self.pretrained == "simplecnn":
            from genperturb.model._simplecnn import SimpleCNN
            self.module.pretrained_model = SimpleCNN()

    def load_tokenizer(self):
        if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
            self.module.tokenizer = None
        elif self.pretrained == "borzoi":
            self.module.tokenizer = None
        elif self.pretrained == "enformer":
            self.module.tokenizer = None
        elif self.pretrained == "simplecnn":
            self.module.tokenizer = None

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

            if self.training_method in ["finetuning", "lora", "prediction", "baseline"]:
                if self.load_lora:
                    model = self.lora_model
                else:
                    model = self.module
                model.cuda()
                model.eval()

                dataloader = JointDataLoaderPred(
                    bed=self.bed,
                    fasta=self.fasta,
                    return_sequence=self.return_sequence,
                    context_length=self.context_length,
                    batch_size=batch_size,
                    mask=mask,
                    tokenizer = self.module.tokenizer,
                    pretrained=self.pretrained,
                )

                for tensors in dataloader.dataloader():
                    out = model.forward(tensors[0].cuda(), cal_loss=False, prediction=True)
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
        model = self.lora_model if self.load_lora else self.module
        model.eval()
        self.module.eval()
        self.module.pretrained_model.eval()

        self.training_method = "prediction"
        self.module.training_method = "prediction"

        out = model.forward(x.cuda(), cal_loss=False, prediction=True)
        return out
