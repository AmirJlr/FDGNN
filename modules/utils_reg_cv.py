import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from torch.optim import Adam
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from copy import deepcopy
from math import sqrt
from tqdm.notebook import tqdm


def run_epoch_reg(model, optimizer, data_loader, loss_function, device, edge_attr, pass_data):
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

        loss = loss_function(pred, data.y.to(torch.float32))

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        y_true.append(data.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    rmse = sqrt(mean_squared_error(y_true, y_pred))

    return np.array(losses).mean(), rmse

def train_reg(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, tensorboard_writer, scheduler):
    writer = SummaryWriter(f'runs/{tensorboard_writer}')

    best_model = None
    best_val_rmse = float('inf')
    best_val_loss = float('inf')
    metrics = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_rmse = run_epoch_reg(model, optimizer, train_loader, loss_function, device, edge_attr, pass_data)
        val_loss, val_rmse = run_epoch_reg(model, None, val_loader, loss_function, device, edge_attr, pass_data)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('RMSE/Train', train_rmse, epoch)
        writer.add_scalar('RMSE/Validation', val_rmse, epoch)

        # Log histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}/values', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}/gradients', param.grad, epoch)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Val loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_loss = val_loss
            best_model = deepcopy(model)

        if scheduler is not None:
            scheduler.step()
        
        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
        })

    writer.close()

    return {
        'best_model': best_model,
        'best_val_loss': best_val_loss,
        'best_val_rmse': best_val_rmse,
        'metrics': metrics
    }


def cross_validate_reg(model, optimizer, scheduler, loss_function, dataset, k_folds=5, num_epochs=75, batch_size=32, device='cpu', edge_attr=False, pass_data=False, tensorboard_writer='TB'):
    results = []

    kfold = KFold(n_splits=k_folds, shuffle=True)

    print('--------------------------------')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(range(len(dataset)))):

        print(f'FOLD {fold}')
        print('--------------------------------')

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids), drop_last=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_ids))

        model.reset_parameters()

        fold_writer = f'{tensorboard_writer}/fold_{fold}'
        result = train_reg(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, fold_writer, scheduler)

        results.append(result)

        print(f'Best RMSE for fold {fold}: {result["best_val_rmse"]:.4f}')

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    final_rmse = sum([res['best_val_rmse'] for res in results]) / len(results)
    print(f'Average RMSE: {final_rmse:.4f}')
    return results, final_rmse
