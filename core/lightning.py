from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch

from .net import TwoLayerReLU
from torch import nn


class System(pl.LightningModule):
    def __init__(self, lr=1e-2, weight_decay=1e-1, **kwargs: Any) -> None:
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = TwoLayerReLU(**kwargs)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return y_hat
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return x, y, y_hat
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5, verbose=True)

        return {
            'optimizer': optim,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

