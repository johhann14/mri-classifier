from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch
from src.data.make_datasets import compute_mean_std

def get_transform(data_dir, batch_size, size):
    mean, std = compute_mean_std(data_dir=data_dir, size=size, batch_size=batch_size)


    tf = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)]
        )
    return tf


def get_dataloaders(train_dir, test_dir, batch_size, size, seed):
    train_tf = get_transform(train_dir, batch_size, size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=train_tf)

    size_train = int(0.8 * len(train_dataset))
    size_val = len(train_dataset) - size_train
    train_dataset, val_dataset = random_split(dataset=train_dataset,
                                              lengths=[size_train, size_val], 
                                              generator=torch.Generator().manual_seed(seed))
    # need to learn about worker
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

