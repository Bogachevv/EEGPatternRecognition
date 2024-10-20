import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

from losses import HybridLoss

import wandb

def _train_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.optimizer.Optimizer,
    criterion: HybridLoss,
    is_binary: bool,
    device: torch.device,
):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    if is_binary:
        running_ones = 0
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        inputs_size = inputs.size(0)

        outputs = model(inputs)
        loss = criterion(input=outputs, target=labels, adj=model.adj.adj_mat)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            _, true_y = torch.max(labels.data, 1)

            if is_binary:
                P = torch.sum(preds)
                N = torch.sum(1 - preds)
                TP = torch.sum(torch.masked_select(true_y, preds == 1))
                TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
                FP = P - TP
                FN = N - TN

                running_ones += P
                running_TP += TP
                running_TN += TN
                running_FP += FP
                running_FN += FN
        
            running_loss += loss.detach() * inputs_size
            running_corrects += torch.sum(preds == true_y)

    running_loss = running_loss.item()
    running_corrects = running_corrects.item()
    if is_binary:
        running_ones = running_ones.item()
        running_TP = running_TP.item()
        running_TN = running_TN.item()
        running_FP = running_FP.item()
        running_FN = running_FN.item()

        epoch_ones = running_ones.double() / (len(train_dataloader.dataset)  // train_dataloader.batch_size)
        epoch_precision = running_TP.double() / (running_TP + running_FP) if running_TP + running_FP != 0 else torch.tensor(0)
        epoch_recall = running_TP.double() / (running_TP + running_FN) if running_TP + running_FN != 0 else torch.tensor(0)
        epoch_f1 = (2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)) if epoch_precision + epoch_recall != 0 else torch.tensor(0)
        epoch_bc = (epoch_recall + running_TN.double() / (running_TN + running_FP)) / 2

    epoch_loss = running_loss / len(train_dataloader.dataset) 
    epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

    # min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(train_dataloader.dataset), 0.05)

    logs_data = {
        'train/loss': epoch_loss,
        # 'train/min_acc': min_acc,
        # 'train/max_acc': max_acc,
        'train/epoch_acc': epoch_acc,
    }
    if is_binary:
        logs_data['train/epoch_bc'] = epoch_bc
        logs_data['train/epoch_ones'] = epoch_ones
        logs_data['train/epoch_precision'] = epoch_precision
        logs_data['train/epoch_recall'] = epoch_recall
        logs_data['train/epoch_f1'] = epoch_f1

        logs_data['train/TP'] = running_TP
        logs_data['train/TN'] = running_TN
        logs_data['train/FP'] = running_FP
        logs_data['train/FN'] = running_FN

    if hasattr(model, 'adj') and hasattr(model.adj.adj_mat, 'requires_grad') and model.adj.adj_mat.requires_grad:
        A_grad = model.adj.adj_mat.grad.detach()
        logs_data['train/A_grad'] = torch.linalg.norm(A_grad)

    wandb.log(data=logs_data)


@torch.inference_mode
def _val_epoch(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: HybridLoss,
    is_binary: bool,
    device: torch.device,
):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    if is_binary:
        running_ones = 0
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_size = inputs.size(0)

        outputs = model(inputs)
        loss = criterion(input=outputs, target=labels, adj=model.adj.adj_mat)

        _, preds = torch.max(outputs, 1)
        _, true_y = torch.max(labels.data, 1)

        if is_binary:
            P = torch.sum(preds)
            N = torch.sum(1 - preds)
            TP = torch.sum(torch.masked_select(true_y, preds == 1))
            TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
            FP = P - TP
            FN = N - TN

            running_ones += P
            running_TP += TP
            running_TN += TN
            running_FP += FP
            running_FN += FN
        
        running_loss += loss.item() * inputs_size
        running_corrects += torch.sum(preds == true_y)

    running_loss = running_loss.item()
    running_corrects = running_corrects.item()
    epoch_loss = running_loss / len(val_dataloader.dataset) 
    epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

    if is_binary:
        running_ones = running_ones.item()
        running_TP = running_TP.item()
        running_TN = running_TN.item()
        running_FP = running_FP.item()
        running_FN = running_FN.item()

        epoch_ones = running_ones.double() / (len(val_dataloader.dataset)  // val_dataloader.batch_size)
        epoch_precision = running_TP.double() / (running_TP + running_FP) if running_TP + running_FP != 0 else torch.tensor(0)
        epoch_recall = running_TP.double() / (running_TP + running_FN) if running_TP + running_FN != 0 else torch.tensor(0)
        epoch_f1 = (2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)) if epoch_precision + epoch_recall != 0 else torch.tensor(0)
        epoch_bc = (epoch_recall + running_TN.double() / (running_TN + running_FP)) / 2

    # min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(val_dataloader.dataset), 0.05)

    logs_data = {
        'val/loss': epoch_loss,
        # 'val/min_acc': min_acc,
        'val/epoch_acc': epoch_acc,
        # 'val/max_acc': max_acc,
    }
    if is_binary:
        logs_data['val/epoch_bc'] = epoch_bc
        logs_data['val/epoch_ones'] = epoch_ones
        logs_data['val/epoch_precision'] = epoch_precision
        logs_data['val/epoch_recall'] = epoch_recall
        logs_data['val/epoch_f1'] = epoch_f1

        logs_data['train/TP'] = running_TP
        logs_data['train/TN'] = running_TN
        logs_data['train/FP'] = running_FP
        logs_data['train/FN'] = running_FN

    wandb.log(data=logs_data)


def train_model():
    pass