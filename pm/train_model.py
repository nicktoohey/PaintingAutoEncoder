import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pm.model.model import PaintingAutoencoder

def train_model(train_loader, val_loader, test_loader, max_epochs=10):
    model = PaintingAutoencoder(lr=1e-5, l1_reg=1e-1, weight_decay=1e-4)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename="best_autoencoder",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model