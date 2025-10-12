import torch

def epoch_optimization(model, device, criterion, optimizer, loader):
    
    model.train()
    running_loss = 0
    n = 0
    total = 0
    correct = 0

    for feat, label in loader:

        feat = feat.to(device)
        label = label.to(device, dtype=torch.long) # important selon la loss (verifier en détail)

        optimizer.zero_grad()

        output = model(feat)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss+= loss

        _, predicted = torch.max(output, dim=1) # index of the wining class for each sample of the batch!!!!!!! -> (batch_size,)
        correct+= (predicted == label).sum().item() # label : (batch_size,)
        n+=1
        total+= label.size(0)
        

    running_loss/= n
    acc = correct/total


    return running_loss, acc


@torch.no_grad()
def epoch_validation(model, device, criterion, loader):
    
    model.eval()

    running_loss = 0
    n = 0
    total = 0
    correct = 0

    for feat, label in loader:
        feat = feat.to(device)
        label = label.to(device, dtype=torch.long) # important selon la loss (verifier en détail)

        output = model(feat)
        loss = criterion(output, label)
        running_loss+= loss

        _, predicted = torch.max(output, dim=1) # index of the wining class for each sample of the batch!!!!!!! -> (batch_size,)
        correct+= (predicted == label).sum().item() # label : (batch_size,)
        n+=1
        total+= label.size(0)


    running_loss/= n
    acc = correct/total
    return running_loss, acc


def training_loop(n_epochs, model, device, criterion, optimizer, train_loader, val_loader):
    for epoch in range(n_epochs):
        print(f"epoch number {epoch}")
        train_loss, train_acc = epoch_optimization(model, device, criterion, optimizer, train_loader)
        val_loss, val_acc = epoch_validation(model, device, criterion, val_loader)
        print(f"\t Training Loss : {train_loss}")
        print(f"\t Training Acc: {train_acc}")
        print(f"\t Validation Loss : {val_loss}")
        print(f"\t Validation Acc: {val_acc}")
        print(f"\n")
