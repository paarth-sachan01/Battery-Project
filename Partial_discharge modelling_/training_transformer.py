import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from data_loading import data_module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, callbacks
from Models.lstm_model import LSTMModel
from Models.tranformer import TransformerModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna import trial
from optuna.samplers import TPESampler

checkpoint_callback = callbacks.ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # Minimize validation loss
    save_top_k=1,        # Save the top 1 best model
    dirpath='checkpoints',  # Directory to save checkpoints
    filename='best_model',  # Checkpoint filename
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor validation accuracy
    mode='min',         # Minimize the monitored metric
    patience=500,         # Number of epochs with no improvement before stopping
    verbose=False,        # Print early stopping updates
)
logger_tensorboard = TensorBoardLogger(save_dir="logs", name="my_lstm_model_equal_data_split") 

def objective(trial):
    # Create a new instance of the model with suggested hyperparameters
    num_heads_suggest = trial.suggest_int('num_head_suggest',2,4,2)
    num_layers_suggest = trial.suggest_int('num_layers_suggest',2,6,1)
    lr_suggest = trial.suggest_loguniform('lr_suggest', 1e-5, 1e-2)
    model = TransformerModel(
        input_size=4,
        num_heads=num_heads_suggest,
        num_layers=num_layers_suggest,
        lr=lr_suggest
    )

    # Define the Lightning trainer with appropriate callbacks
    trainer = pl.Trainer(
        logger=True,
        max_epochs=50,
        accelerator="auto",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
        deterministic=True
    )

    hyperparameters = dict(num_heads=num_heads_suggest,num_layers=num_layers_suggest,
                           lr=lr_suggest)
    trainer.logger.log_hyperparams(hyperparameters)


    # Fit the model
    trainer.fit(model, datamodule=data_module)

    # Get the validation loss from the best model checkpoint
    #checkpoint = torch.load(checkpoint_callback.best_model_path)
    val_loss = model.best_val_loss

    return val_loss


study = optuna.create_study(direction='minimize',pruner=optuna.pruners.NopPruner()
                            ,sampler=TPESampler())  # We want to minimize the validation loss

# Create a PyTorch Lightning callback for pruning


# Run the optimization
study.optimize(objective, n_trials=8)
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_hyperparameters = study.best_params
num_heads = best_hyperparameters['num_head_suggest']
num_layers = best_hyperparameters['num_layers_suggest']
learning_rate = best_hyperparameters['lr_suggest']

best_model = LSTMModel(
    input_size=4,
    num_heads=num_heads,
    num_layers=num_layers,
    lr=learning_rate
)

trainer_ = pl.Trainer(
    logger=logger_tensorboard,
    max_epochs=50000,
    accelerator="auto",
    callbacks=checkpoint_callback,
    deterministic=True
)

trainer_.fit(best_model, datamodule=data_module)

trainer_.test(best_model, datamodule=data_module)

