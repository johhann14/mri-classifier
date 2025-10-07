from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch

BATCH_SIZE = 32
SIZE = (150,150)
SEED = 42
N_CLASSES = 4

def compute_mean_std(data_dir, size=SIZE, batch_size=BATCH_SIZE):
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



    
# taken from Sovit Ranjan Rath
def save_model(epochs, model, optimize, criterion):
    return

    
