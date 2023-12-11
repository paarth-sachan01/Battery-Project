import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transfer_learning_dataloading import transfer_learning_data_module
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
from pytorch_lightning.tuner import Tuner

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
logger_tensorboard = TensorBoardLogger(save_dir="logs_transfer_learning", 
                                       name="my_lstm_model_equal_data_split") 

path ="/Users/paarthsachan/technical/State_of_health_battery/Partial_discharge_modelling/checkpoints/best_model-v3.ckpt"
best_model = LSTMModel.load_from_checkpoint(path,
                        input_size=4,  hidden_size=48, num_layers=1,  output_size=1)

trainer = pl.Trainer(max_epochs=500,accelerator="auto", callbacks=checkpoint_callback,
                     deterministic=True)

tuner = Tuner(trainer)

lr_finder = tuner.lr_find(best_model, datamodule=transfer_learning_data_module,
                          num_training=500)
best_model.learning_rate=lr_finder.suggestion()
# best_model.learning_rate= 4e-5
trainer = pl.Trainer(
    logger=logger_tensorboard,
    max_epochs=17500,
    accelerator="auto",
    #callbacks=checkpoint_callback,
    deterministic=True,
)

trainer.fit(best_model, transfer_learning_data_module)
#trainer.test(best_model, datamodule=transfer_learning_data_module)




