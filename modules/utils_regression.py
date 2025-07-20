import numpy as np
import torch
from torch import device
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from copy import deepcopy
from math import sqrt 
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os


def run_epoch_reg(model, optimizer, data_loader, loss_function, device, edge_attr, pass_data):

    model.to(device)
    model.train() if optimizer is not None else model.eval()

    y_true = []
    y_pred = []
    losses = []

    for step, data in enumerate(tqdm(data_loader, desc="Iteration")):
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
        
        loss = loss_function(pred, data.y)  # Calculate loss

        if optimizer is not None:
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

        losses.append(loss.detach().cpu().numpy())
        y_true.append(data.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Calculate RMSE using predicted and true values
    rmse = sqrt(((y_true - y_pred) ** 2).mean())

    return np.array(losses).mean(), rmse



def train_reg(model, optimizer,loss_function, train_loader, val_loader, num_epochs, device, edge_attr, pass_data, tensorboard_writer):
   
    writer = SummaryWriter(f'runs/{tensorboard_writer}')  # More descriptive run name

    best_model = None
    best_val_rmse = float('inf')  # Initialize with positive infinity for minimization

    for epoch in range(1, num_epochs + 1):
        train_loss, train_rmse = run_epoch_reg(model, optimizer, train_loader,loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('rmse/train', train_rmse, epoch)

        val_loss, val_rmse = run_epoch_reg(model, None, val_loader, loss_function, device, edge_attr, pass_data)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('rmse/val', val_rmse, epoch)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Val loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

        if val_rmse < best_val_rmse:  # Track best model based on lowest validation RMSE
            best_val_rmse = val_rmse
            best_model = deepcopy(model)

    writer.close()  # Close TensorBoard writer
    return {
        'best_model': best_model,
        'best_val_rmse': best_val_rmse
    }


# # After training
# results = train_reg(model, optimizer, loss_function, train_data, val_data, num_epochs, device, edge_attr, pass_data, tensorboard_writer)

# best_model = results['best_model']
# best_val_rmse = results['best_val_rmse']

# print(f"Best validation RMSE: {best_val_rmse:.4f}")

# # Save the best model
# torch.save(best_model.state_dict(), 'best_model.pth')

# # To load the model later
# # Instantiate the model class first (ensure the model class is defined the same way)
# model = YourModelClass()
# model.load_state_dict(torch.load('best_model.pth'))
# model.to(device)