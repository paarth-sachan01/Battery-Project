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


class Grad_clipping(Callback):
    def on_before_optimizer_step(self, trainer, pl_module,optimizer):
       torch.nn.utils.clip_grad_norm_(pl_module.parameters(), max_norm=0.3)
    
model =TransformerModel(input_size=4, hidden_size=64, 
                        num_layers=4, output_size=1,num_heads=2)


# model = LSTMModel(input_size=4, hidden_size=64, num_layers=1, output_size=1,lr=4e-4)
logger = TensorBoardLogger(save_dir="logs", name="my_lstm_model_equal_data_split") 
trainer = pl.Trainer(max_epochs=50000,accelerator="auto",logger=logger, 
                     callbacks=checkpoint_callback,
                     deterministic=True)
# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(model, datamodule=data_module)
# model.learning_rate=lr_finder.suggestion()

trainer.fit(model, data_module)
# best_model = LSTMModel.load_from_checkpoint(checkpoint_callback.best_model_path,
#                         input_size=4,  hidden_size=64, num_layers=1,  output_size=1)

best_model = TransformerModel.load_from_checkpoint(checkpoint_callback.best_model_path,
                        input_size=4, hidden_size=64, 
                        num_layers=4, output_size=1,num_heads=2)


# Test the best model with the data module
trainer.test(best_model, datamodule=data_module)


