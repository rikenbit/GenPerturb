import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR


class Training(pl.LightningModule):
    def __init__(
        self, 
        module: nn.Module,
        use_device: str = "cpu",
        lr: float = 1e-4,
        strategy: str = None,
        warmup: bool = False,
        warmup_steps: int = 47, 
        usage_print_every: int = 200,
        freeze_gate_step: int = -1,
        alpha_floor_eps: float = 0.05,
    ):
        super().__init__()

        self.module = module
        self.use_device = use_device
        self.lr = lr
        self.strategy = strategy
        self.warmup = warmup
        self.warmup_steps = warmup_steps

        self.usage_print_every = usage_print_every
        self.freeze_gate_step = freeze_gate_step

    def forward(self, x):
        return self.module(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.005)

        if self.warmup:
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                else:
                    return 1.0
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", 
                    "frequency": 1,
                }
            }
        else:
            return optimizer

    def training_step(self, batch, batch_index):
        loss = self.forward(batch)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_index):
        loss = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
            hasattr(self.module, "report_expert_usage") and
            self.usage_print_every > 0 and
            (self.global_step % self.usage_print_every == 0)
        ):
            self.module.report_expert_usage(reset=True)

        if self.freeze_gate_step >= 0 and self.global_step == self.freeze_gate_step:
            if hasattr(self.module, "gate"):
                for p in self.module.gate.parameters():
                    p.requires_grad = False
                print(f"[INFO] gate frozen at global_step {self.global_step}")


class LoadTrainedModule(pl.LightningModule):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()

        self.module = module

    def forward(self, x, cal_loss=False, prediction=False):
        return self.module(x, cal_loss=cal_loss, prediction=prediction)



