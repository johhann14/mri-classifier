import pandas as pd
import os
import torch 

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def create_inital_df(base_dir):
    data = []

    for split in ['Training', 'Testing']:
        split_dir = os.path.join(base_dir, split)

        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for img in os.listdir(label_dir):
                    img_path = os.path.abspath(os.path.join(label_dir, img))
                    data.append({
                        'path': img_path,
                        'label': label,
                        'split': split.lower()
                    })

    return pd.DataFrame(data)

def compute_mean_std(data_dir, size, batch_size):
    # input has 3 channels
    transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor()]) # to tensor : HxWxC -> CxHxW
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0
    # mean per batch -> each batch impacts the global mean the same way = no bias 
    # bc last batch can be smaller
    for feat, _ in loader:
        bs = feat.size(0)

        batch_mean = feat.mean(dim=(0,2,3)) # on every batch on H x W
        batch_std = feat.std(dim=(0,2,3))

        mean += batch_mean * bs
        std+= batch_std * bs
        total += bs

    mean /= total
    std /= total

    return mean, std