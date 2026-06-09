
from pm.dataset.dataloader import get_data_loaders
from pm.model.model import PaintingAutoencoder
from pm.train_model import train_model

# Test data loading
train_loader, val_loader, test_loader = get_data_loaders(base_path='./data', name='all', batch_size=4)

# Check a batch loads correctly
imgs, labels = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}")       # should be (4, 3, 512, 512)
print(f"Labels: {labels}")
assert imgs.shape[1] == 3, "Expected 3 channels"
assert imgs.min() >= -1 and imgs.max() <= 1, "Expected normalized to [-1, 1]"
print("Data loading OK")

# Test model forward pass
model = PaintingAutoencoder()
output, latent = model(imgs)
print(f"Output shape: {output.shape}")    # should match input
print(f"Latent shape: {latent.shape}")
print("Model forward pass OK")

# Test training (1 epoch, small data)
train_model(train_loader, val_loader, test_loader, max_epochs=1)
print("Training OK")