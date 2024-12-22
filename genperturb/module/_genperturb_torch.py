import torch
from torch import nn
import polars as pl
from typing import Dict, List, Optional, Union


class GenPerturbTorch(nn.Module):
    def __init__(
        self,
        num_perturb=None,
        context_length : int = 64_128,
        pretrained : Optional[str] = "enformer",
        emb_method : Optional[str] = "tss",
        layer : int = 1,
        target_length : int = 4,
        training_method : Optional[str] = "transfer",
    ):
        super().__init__()

        self.num_perturb    = num_perturb
        self.context_length = context_length
        self.pretrained     = pretrained
        self.emb_method     = emb_method
        self.target_length = target_length
        self.training_method = training_method

        if self.pretrained == "enformer":
            print("load : enformer")
            emb_dim = 3072 * self.target_length

        elif self.pretrained in ["hyena_dna_tss", "hyena_dna_last", "hyena_dna_mean"]:
            print("load : hyena_dna")
            if emb_method == "tss":
                emb_dim = 256 * 512
                layer = 2
            elif emb_method == "last":
                emb_dim = 256 
            elif emb_method == "mean":
                emb_dim = 256

        elif self.pretrained in ["nucleotide_transformer_tss", "nucleotide_transformer_cls", "nucleotide_transformer_mean"]:
            print("load : nucleotide_transformer")
            if emb_method == "tss":
                emb_dim = 1024 * 86 #516bp
                layer = 2
            elif emb_method == "cls":
                emb_dim = 1024
            elif emb_method == "mean":
                emb_dim = 1024

        if layer == 2:
            n_hidden = 3072
            self._heads = nn.Sequential(
                nn.Linear(emb_dim, n_hidden),
                nn.InstanceNorm1d(n_hidden, momentum=0.1, eps=0.0001),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(n_hidden, self.num_perturb),
                nn.ReLU()
            )
        else:
            self._heads = nn.Sequential(
                nn.Linear(emb_dim, self.num_perturb),
                nn.ReLU()
            )

    def mse_loss(self, pred, target):
        return nn.MSELoss(reduction="none")(pred, target).mean()
    
    def weighted_mse_loss(self, pred, target):
        weight = 1 / torch.abs(target).sum(1)
        return (((pred - target) ** 2) * weight.unsqueeze(1)).mean()

    def poisson_loss(self, pred, target):
        log_pred = torch.log(pred.clamp(min = 1e-20))
        return (pred - target * log_pred).mean()

    def forward(
            self,
            xs,
            cal_loss = True,
            prediction = False,
        ):

        if prediction:
            input_seq = xs
        else:
            input_seq = xs[0]

        if self.training_method in ["prediction", "finetuning", "lora"]:
            if self.pretrained == "enformer":
                if input_seq.dim() == 2:
                    emb = self.pretrained_model(input_seq, return_only_embeddings=True).reshape(1, -1)
                elif input_seq.dim() == 3:
                    emb = self.pretrained_model(input_seq, return_only_embeddings=True).flatten(start_dim=1)

            elif self.pretrained in ["hyena_dna_tss", "hyena_dna_last", "hyena_dna_mean"]:
                print("## prediction/finetuning/lora ##")

                emb = self.pretrained_model(input_seq, output_hidden_states=True)['hidden_states'][-1]

                if self.emb_method == "tss":
                    emb = emb[:,79743:80255,:].flatten(start_dim=1)
                elif self.emb_method == "last":
                    emb = emb[:,159_998,:]
                elif self.emb_method == "mean":
                    emb = torch.sum(emb, axis=-2) / 159_999

            elif self.pretrained in ["nucleotide_transformer_tss", "nucleotide_transformer_cls", "nucleotide_transformer_mean"]:
                emb = self.pretrained_model(input_seq, output_hidden_states=True)['hidden_states'][-1]

                if self.emb_method == "tss":
                    emb = emb[:,982:1068,:].flatten(start_dim=1)
                elif self.emb_method == "cls":
                    emb = emb[:,0,:]
                elif self.emb_method == "mean":
                    emb = torch.sum(emb, axis=-2) / 2048
            
            out = self._heads(emb)

        elif self.training_method == "transfer":
            print("## transfer ##")
            out = self._heads(input_seq)


        if cal_loss:
            loss = self.mse_loss(out, xs[1])
            return loss

        else:
            return out
