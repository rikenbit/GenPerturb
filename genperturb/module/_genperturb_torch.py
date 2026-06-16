import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

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

        if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
            print(f"load : {self.pretrained}")
            emb_dim = 3072 * self.target_length

        elif self.pretrained == "borzoi":
            print("load : borzoi")
            emb_dim = 1920 * 16

        elif self.pretrained == "enformer":
            print("load : enformer")
            emb_dim = 3072 * self.target_length

        elif self.pretrained == "simplecnn":
            print("load : simplecnn")
            emb_dim = 39600


        self._heads = nn.Sequential(
            nn.Linear(emb_dim, self.num_perturb),
            nn.ReLU()
        )

    def mse_loss(self, pred, target):
        return nn.MSELoss(reduction="none")(pred, target).mean()
     
    def mse_loss_with_negative(self, pred, target, alpha: float = 1.0, eps: float = 1e-8):
        mse_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    
        x = pred - pred.mean(dim=-1, keepdim=True)
        y = target - target.mean(dim=-1, keepdim=True)
        cov   = (x * y).mean(dim=-1)
        denom = (x.pow(2).mean(dim=-1).sqrt() * y.pow(2).mean(dim=-1).sqrt()).clamp(min=eps)
        r = cov / denom
    
        penalty = 1.0 + alpha * F.relu(-r)
        return (penalty * mse_per_sample).mean()

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

        if self.training_method in ["prediction", "finetuning", "lora", "baseline"]:

            if self.pretrained == "alphagenome" or self.pretrained.startswith("alphagenome_fold_"):
                if input_seq.dim() == 1:
                    seq_input = input_seq.unsqueeze(0)
                elif input_seq.dim() == 2 and input_seq.shape[-1] == 4 and torch.is_floating_point(input_seq):
                    seq_input = input_seq.unsqueeze(0)
                else:
                    seq_input = input_seq

                if seq_input.dim() == 2:
                    invalid_mask = (seq_input < 0) | (seq_input > 3)
                    if invalid_mask.any():
                        raise ValueError(
                            "AlphaGenome sequence indices must be resolved before model.forward(). "
                            "Use resolve_alphagenome_indices() with an interval-specific key."
                        )
                    seq_input = seq_input.long()

                emb = self.pretrained_model(seq_input)
                emb = emb.flatten(start_dim=1)  # (batch, 4*3072)

            elif self.pretrained == "borzoi":
                if input_seq.dim() == 2:
                    input_seq = input_seq.unsqueeze(0).permute(0, 2, 1)
                elif input_seq.dim() == 3:
                    input_seq = input_seq.permute(0, 2, 1)
                emb = self.pretrained_model(input_seq).flatten(start_dim=1)

            elif self.pretrained == "enformer":
                if input_seq.dim() == 2:
                    emb = self.pretrained_model(input_seq, return_only_embeddings=True).reshape(1, -1)
                elif input_seq.dim() == 3:
                    emb = self.pretrained_model(input_seq, return_only_embeddings=True).flatten(start_dim=1)

            elif self.pretrained == "simplecnn":
                if input_seq.dim() == 2:
                    emb = self.pretrained_model(input_seq.unsqueeze(0)).flatten(start_dim=1)   # (1,L,4) → (1,L)
                elif input_seq.dim() == 3:
                    emb = self.pretrained_model(input_seq).flatten(start_dim=1)  # (B,L,4) → (B,L)

            out = self._heads(emb)

        elif self.training_method == "transfer":
            out = self._heads(input_seq)


        if cal_loss:
            loss = self.mse_loss(out, xs[1])
            #loss = self.weighted_mse_loss(out, xs[1])
            #loss = self.mse_loss_with_negative(out, xs[1])
            return loss

        else:
            return out

