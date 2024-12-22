import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam



class Training(pl.LightningModule):
    def __init__(
        self, 
        module: nn.Module,
        use_device: str  = "cpu",
        lr: float = 1e-4,
        strategy : str = None,
        warmup : bool = False,
    ):
        super().__init__()

        self.module = module
        self.use_device = use_device
        self.lr = lr
        self.strategy = strategy
        self.warmup = warmup

    def forward(self, x):
        return self.module(x)

    def configure_optimizers(self):
        # Modify the optimizer and learning rate by changing the comment out according to the conditions during training.
        if self.strategy is not None:
            if self.strategy == "deepspeed_stage_3_offload" or self.strategy == "deepspeed_stage_2_offload":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
                #optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr, weight_decay=0.005)
            elif self.strategy == "deepspeed_stage_3" or self.strategy == "deepspeed_stage_2" or self.strategy == "ddp":
                #optimizer = FusedAdam(parameters, lr=self.lr, weight_decay=0.005)
                #optimizer = FusedAdam(self.parameters(), lr=self.lr, weight_decay=0.005)
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
                #optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.005)
            else:
                #parameters = [
                #    {'params': self.module.pretrained_model.parameters(), 'lr': 1e-10},
                #    {'params': self.module._heads.parameters(), 'lr': 1e-4},
                #]
                #optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.005)
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
        else:
            #parameters = [
            #    {'params': self.module.pretrained_model.parameters(), 'lr': 1e-10},
            #    {'params': self.module._heads.parameters(), 'lr': 1e-4},
            #]
            #optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.005)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
        return optimizer


    def training_step(self, batch, batch_index):
        loss = self.forward(batch)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_index):
        loss = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        optimizer.step(closure=optimizer_closure)

        if self.warmup == True:
            if self.trainer.global_step < 10000:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / 10000.0)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr

class LoadTrainedModule(pl.LightningModule):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()

        self.module = module

    def forward(self, x, cal_loss=False, prediction=False):
        return self.module(x, cal_loss=cal_loss, prediction=prediction)



