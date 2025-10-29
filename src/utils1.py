from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch
from pathlib import Path
import matplotlib.pyplot as plt

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



    
# taken from Sovit Ranjan Rath
def save_model(path, epochs, model, optimizer, criterion):
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimize_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, Path(path) / "model.pth") 
    return

def save_plots(path, train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 8))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(Path(path) / "accuracy.png")
    
    # loss plots
    plt.figure(figsize=(10, 8))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path(path) / "loss.png")



def show_images(dataframe, n_sample, size, mode="RGB", random_state=None):
    plt.figure(figsize=(22,22))
    shuffle_df = dataframe.sample(n_sample, random_state=random_state)
    for i in range(n_sample):
        path = shuffle_df['path'].iloc[i]
        label = shuffle_df['label'].iloc[i]
        index = shuffle_df.index[i]
        plt.subplot(n_sample//4 + 1, 4, i+1)
        img = Image.open(path).convert(mode).resize(size)
        plt.imshow(img)
        plt.title(str(index) + ": " + label)
    plt.show()
        

    
