from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch

import torchaudio as ta

import pytorch_lightning as pl

class SparseFourierDataset(Dataset):
    def __init__(self, dim=256, n_samples=256, zca=True):

        assert n_samples <= dim, "n_samples must be less than or equal to dim"

        self.dim = dim
        self.n_samples = n_samples
        self.zca = zca

        self.x, self.y = self.generate_data()

    def generate_data(self):

        f = torch.arange(0, self.n_samples, dtype=torch.float32)

        x = torch.cos(2 * torch.pi * f[:, None] * torch.linspace(0.0, 1.0, self.dim)[None, :])

        if self.zca:
            x = self.zca_whiten(x)

        return x, f/self.n_samples
    
    def zca_whiten(self, x):

        _, S, Vh = torch.linalg.svd(x, full_matrices=True)

        x = x @ Vh.T @ torch.diag(1.0 / torch.sqrt(S + 1e-8))

        return x

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]
    
class SparseFourierDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.train_dataset = SparseFourierDataset(**self.kwargs)
        self.val_dataset = SparseFourierDataset(**self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)