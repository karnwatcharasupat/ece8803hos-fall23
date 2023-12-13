import os
import numpy as np
import pytorch_lightning as pl
import torch
from core.data import SparseFourierDataModule

from core.lightning import System

def train(dim=256, n_samples=None, zca=False, batch_size=None, lr=1e-3, weight_decay=1e-1, hidden_dim=None, output_dim=1, bias_first=True, bias_second=True):
   
    if n_samples is None:
        n_samples = dim // 8

    if batch_size is None:
        batch_size = n_samples

    if hidden_dim is None:
        hidden_dim = dim * 4
   
    data = SparseFourierDataModule(
        dim=dim,
        n_samples=n_samples,
        zca=zca,
        batch_size=batch_size
    )
    model = System(
        lr=lr,
        weight_decay=weight_decay,
        input_dim=dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        bias_first=bias_first,
        bias_second=bias_second
    )

    trainer = pl.Trainer(accelerator="gpu", max_epochs=1000, log_every_n_steps=1, callbacks=
                         [
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')
        ])

    trainer.fit(model, data)

    output = trainer.predict(model, data)

    x = [o[0] for o in output]
    
    y = [   o[1]  for o in output ]

    y_hat = [     o[2]     for o in output  ]

    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()
    y_hat = torch.cat(y_hat, dim=0).detach().cpu().numpy()

    np.savez(os.path.join(trainer.logger.log_dir, 'output.npz'), x=x, y=y, y_hat=y_hat)

if __name__ == '__main__':
    import fire
    fire.Fire(train)