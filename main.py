import argparse

import torch
import torch.nn as nn
from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.train import epoch_optimization, epoch_validation, training_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mri-classifier main')
    parser.add_argument('--e', '--epochs', type=int, default=20, help='Number of epochs to train our network for')
    parser.add_argument('--device', type=str, default='cuda', help='Device to compute on')
    parser.add_argument('--seed', type=int, default=None, help='Seed to reproduce experiences')
    parser.add_argument('--method', type=str, default='adam', help='Optimizer ')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of the optimizer')
    parser.add_argument('--bs', '--batch-size',type=int, default=32, help='Batch size')
    commmand_args = parser.parse_args()
    args = vars(commmand_args)
    
    #todo faire la seed 


    train_dir = "data/Training"
    test_dir = "data/Testing"

    train_loader, val_loader, test_loader = get_dataloaders(train_dir=train_dir, test_dir=test_dir, batch_size=args['bs'])


    model = SimpleCNN()
   # model.to(args['device'])
    
    criterion = nn.CrossEntropyLoss()
    if args['method'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    
    print(f"Training loop")
    training_loop(args['e'], model, args['device'], criterion, optimizer, train_loader, test_loader)