import torch
from tqdm.auto import tqdm
from src.metrics_logger import new_run, log_metrics
from src.utils1 import save_model, save_plots

def epoch_optimization(model, device, criterion, optimizer, loader):
    print('Training')
    model.train()
    running_loss = 0
    n = 0
    total = 0
    correct = 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        feat, label = data
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


    return running_loss.item(), acc


@torch.no_grad()
def epoch_validation(model, device, criterion, loader):
    print('Validation')
    model.eval()

    running_loss = 0
    n = 0
    total = 0
    correct = 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        feat, label = data
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
    return running_loss.item(), acc


def training_loop(n_epochs, model, device, criterion, optimizer, train_loader, val_loader, root=None):
    
    run_dir, mcsv = new_run(root=root)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(1, n_epochs + 1):
        print(f"[INFO]: Epoch {epoch}/{n_epochs}")
        train_epoch_loss, train_epoch_acc = epoch_optimization(model, device, criterion, optimizer, train_loader)
        val_epoch_loss, val_epoch_acc = epoch_validation(model, device, criterion, val_loader)
        log_metrics(mcsv, split="train", epoch=epoch, loss=train_epoch_loss, acc=train_epoch_acc)
        log_metrics(mcsv, split="val", epoch=epoch, loss=val_epoch_loss, acc=val_epoch_acc)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        print(f"\t Training Loss : {train_epoch_loss:.3f}, Training acc: {train_epoch_acc:.3f}")
        print(f"\t Validation Loss : {val_epoch_loss:.3f}, Validation Acc: {val_epoch_acc:.3f}")
        print(f"------------------------------------")
    save_plots(path=run_dir, train_acc=train_acc, valid_acc=val_acc, train_loss=train_loss, valid_loss=val_loss)
    save_model(path=run_dir, epochs=10, model=model, optimizer=optimizer, criterion=criterion)

    return train_acc, val_acc, train_loss, val_loss
