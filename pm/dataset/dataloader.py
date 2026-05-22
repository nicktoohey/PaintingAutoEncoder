import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

CLASS_NAMES = ['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 'Monet','Renoir', 'Sargent', 'VanGogh']

#Might change depending on the model
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class PaintingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(img_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img in os.listdir(class_dir):
                if img.endswith('.jpg'):
                    self.images.append(os.path.join(class_dir, img))
                    self.labels.append(class_name)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(base_path='./',name='example', batch_size=32):
    '''Create data loaders for train, test, and dev sets

    Args:
        base_path (string): Base path to the dataset
        name (string): One of all,example, test,train,(develop,validation)
        batch_size (int): Batch size for the dataloaders
        
    Notes:
        Train and test images are loaded relative to ``base_path``
    '''
    if name =='example':
        base_path = os.path.join(os.path.dirname(__file__), "..","..", "tests", "example_dataset")
        example_dataset = PaintingDataset(
            img_dir=base_path,
            transform=DEFAULT_TRANSFORM,
        )
        example_loader = DataLoader(
            example_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        return example_loader
    
    if name in ['train', 'all']:
        train_dataset = PaintingDataset(
            img_dir=os.path.join(base_path,"training", "training"),
            transform=DEFAULT_TRANSFORM,
       
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
    if name == 'train': return train_loader

    if name in ['test', 'validation', 'all']:
        full_val_dataset = PaintingDataset(
            img_dir=os.path.join(base_path, "validation", "validation"),
            transform=DEFAULT_TRANSFORM,
        )
        val_size  = len(full_val_dataset) // 2
        test_size = len(full_val_dataset) - val_size

        val_dataset, test_dataset = random_split(
            full_val_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

    if name == 'validation': return val_loader
    if name == 'test': return test_loader

    return train_loader, test_loader, val_loader