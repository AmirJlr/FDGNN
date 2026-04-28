import modules.optuna as optuna
import torch

def objective(trial):
    # Suggest hyperparameters
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    heads = trial.suggest_categorical('heads', [2, 4, 6, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # Build model
    model = GINGAT(
        node_dim=9,
        edge_dim=3,
        hidden_channels=hidden_channels,
        out_channels=N_COMPONENTS,
        heads=heads, 
        dropout=dropout,
        pooling_type='gru',
        num_tasks=2,
        use_dummy=True,
        feature_mode='both',
        num_gin_layers=4,
        num_gat_layers=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    results = train_multi_cls(
        model=model,
        optimizer=optimizer,
        loss_function=LOSS_FUNCTION,
        train_loader=train_loader,
        val_loader=valid_loader,
        num_epochs=EPOCHS,
        device=device,
        edge_attr=True,
        pass_data=True,
        tensorboard_writer=f"optuna_trial_{trial.number}"
    )

    best_model = results['best_model']

    # Evaluate on validation set
    _, val_auc = run_epoch_multi_cls(
        model=best_model, 
        optimizer=None, 
        data_loader=valid_loader,
        loss_function=LOSS_FUNCTION, 
        device=device, 
        edge_attr=True, 
        pass_data=True
    )

    # Clean up memory
    del model, optimizer, best_model
    torch.cuda.empty_cache()

    return val_auc


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n" + "="*50)
print("Best trial:")
print(f"  Validation AUC: {study.best_trial.value:.4f}")
print("  Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best hyperparameters
print("\n" + "="*50)
print("Training final model with best hyperparameters...")

best_params = study.best_trial.params
final_model = GINGAT(
    node_dim=9,
    edge_dim=3,
    hidden_channels=best_params['hidden_channels'],
    out_channels=N_COMPONENTS,
    heads=best_params['heads'], 
    dropout=best_params['dropout'],
    pooling_type='gru',
    num_tasks=2,
    use_dummy=True,
    feature_mode='both',
    num_gin_layers=4,
    num_gat_layers=1
).to(device)

final_optimizer = torch.optim.Adam(
    final_model.parameters(), 
    lr=best_params['lr'], 
    weight_decay=best_params['weight_decay']
)

final_results = train_multi_cls(
    model=final_model,
    optimizer=final_optimizer,
    loss_function=LOSS_FUNCTION,
    train_loader=train_loader,
    val_loader=valid_loader,
    num_epochs=EPOCHS,
    device=device,
    edge_attr=True,
    pass_data=True,
    tensorboard_writer="final_best_model"
)

# Test evaluation
best_final_model = final_results['best_model']
_, test_auc = run_epoch_multi_cls(
    model=best_final_model,
    optimizer=None,
    data_loader=test_loader,
    loss_function=LOSS_FUNCTION,
    device=device,
    edge_attr=True,
    pass_data=True
)

print(f"\nFinal Test AUC: {test_auc:.4f}")

# Save results
import pandas as pd
from datetime import datetime

results_df = pd.DataFrame([{
    'hidden_channels': best_params['hidden_channels'],
    'heads': best_params['heads'],
    'dropout': best_params['dropout'],
    'lr': best_params['lr'],
    'weight_decay': best_params['weight_decay'],
    'val_auc': study.best_trial.value,
    'test_auc': test_auc,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}])

results_df.to_csv('optuna_best_results.csv', index=False)
print("\nResults saved to optuna_best_results.csv")