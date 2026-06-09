import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class PaintingAutoencoder(pl.LightningModule):
    def __init__(self, lr=1e-5, l1_reg=1e-1, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.criterion = nn.MSELoss()

    def forward(self, x):
        latent = self.autoencoder.encode(x).latent_dist.sample()
        output = self.autoencoder.decode(latent).sample
        return output, latent

    def _shared_step(self, batch):
        img, _ = batch
        output, latent = self(img)
        loss = self.criterion(output, img) + latent.abs().mean() * self.hparams.l1_reg
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer