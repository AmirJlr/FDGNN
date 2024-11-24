import os
import numpy as np
import torch
import torch.nn as nn
from torch import device
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.optim import Adam


from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from copy import deepcopy
from math import sqrt 
from tqdm.notebook import tqdm


def run_epoch_cls(model, optimizer, data_loader, loss_function, device, edge_attr, pass_data):
    """
    Runs a single training epoch for a PyG model on a graph property prediction task.

    Args:
        model (torch.nn.Module): The PyG model to be trained.
        optimizer (torch.optim.Optimizer, optional): The optimizer for training. Defaults to None.
        data_loader (torch_geometric.data.DataLoader): The data loader for the training data.
        loss_function (torch.nn.Module, optional): The loss function to use. Defaults to BCEWithLogitsLoss().
        device (str, optional): The device to use for training ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        tuple: A tuple containing the average loss and ROC-AUC score for the epoch.
    """

    model.to(device)
    model.train() if optimizer is not None else model.eval()

    y_true = []
    y_pred = []
    losses = []

    for step, data in enumerate(tqdm(data_loader, desc="Iteration")):  # Iterate in batches over the training dataset.
        data = data.to(device)  # Move data batch to device

        if edge_attr :
            if pass_data :
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch, data)
            else :
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        else :
            if pass_data :
                pred = model(data.x, data.edge_index, data.batch, data)
            else :
                pred = model(data.x, data.edge_index, data.batch)

        loss = loss_function(pred, data.y.to(torch.float32))  # Calculate loss

        if optimizer is not None:
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

        losses.append(loss.detach().cpu().numpy())
        y_true.append(data.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Calculate ROC-AUC score using sklearn
    auc_roc = roc_auc_score(y_true, y_pred)

    return np.array(losses).mean(), auc_roc



def train_cls(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, tensorboard_writer):

    writer = SummaryWriter(f'runs/{tensorboard_writer}')

    best_model = None
    best_val_auc = 0
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc = run_epoch_cls(model, optimizer, train_loader,loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('auc/train', train_auc, epoch)

        val_loss, val_auc = run_epoch_cls(model, None, val_loader,loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('auc/val', val_auc, epoch)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ROC-AUC: {train_auc:.4f}, Val loss: {val_loss:.4f}, Val ROC-AUC: {val_auc:.4f}')

        if val_auc > best_val_auc : 
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_model = deepcopy(model)

    writer.close()  # Close TensorBoard writer
    return {
        'best_model': best_model,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc
    }


# results = train_cls(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, tensorboard_writer)
# best_model = results['best_model']
# best_val_rmse = results['best_val_rmse']

# # Save the best model
# torch.save(best_model.state_dict(), 'best_model.pth')

# # To load the model later
# # Instantiate the model class first (ensure the model class is defined the same way)
# model = YourModelClass()
# model.load_state_dict(torch.load('best_model.pth'))
# model.to(device)



######### Multi Task Classification #########

def multi_task_loss(pred, target, loss_function):
    mask = ~torch.isnan(target)
    if mask.any():
        loss = loss_function(pred[mask], target[mask].to(torch.float32))
        return loss.mean()  # Reduce the loss across the batch
    return torch.tensor(0.0, requires_grad=True)


def run_epoch_multi_cls(model, optimizer, data_loader, loss_function, device, edge_attr, pass_data):
    model.to(device)
    model.train() if optimizer is not None else model.eval()

    y_true = []
    y_pred = []
    losses = []

    for step, data in enumerate(tqdm(data_loader, desc="Iteration")):
        data = data.to(device)

        if edge_attr:
            if pass_data:
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch, data)
            else:
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        else:
            if pass_data:
                pred = model(data.x, data.edge_index, data.batch, data)
            else:
                pred = model(data.x, data.edge_index, data.batch)

        loss = multi_task_loss(pred, data.y, loss_function)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        y_true.append(data.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Calculate ROC-AUC score for each task using sklearn
    auc_roc = []
    for i in range(y_true.shape[1]):
        valid_indices = ~np.isnan(y_true[:, i])
        if valid_indices.any():
            try:
                auc_roc.append(roc_auc_score(y_true[valid_indices, i], y_pred[valid_indices, i]))
            except ValueError as e:
                print(f"Error calculating ROC AUC for task {i}: {e}")
                auc_roc.append(np.nan)
        else:
            auc_roc.append(np.nan)

    avg_auc_roc = np.nanmean(auc_roc)

    return np.array(losses).mean(), avg_auc_roc



def train_multi_cls(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, tensorboard_writer):

    writer = SummaryWriter(f'runs/{tensorboard_writer}')

    best_model = None
    best_val_auc = 0
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc = run_epoch_multi_cls(model, optimizer, train_loader,loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('auc/train', train_auc, epoch)

        val_loss, val_auc = run_epoch_multi_cls(model, None, val_loader,loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('auc/val', val_auc, epoch)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ROC-AUC: {train_auc:.4f}, Val loss: {val_loss:.4f}, Val ROC-AUC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_model = deepcopy(model)

    writer.close()  # Close TensorBoard writer
    
    return {
        'best_model': best_model,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc
    }