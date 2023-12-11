import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from data_loading import data_module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class TransformerModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size,num_heads=2):
        super(TransformerModel, self).__init__()

        #self.embedding= nn.Embedding(input_size, 1)

        #self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4,batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4,batch_first=True)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=32)
        self.layer_norm = nn.LayerNorm(input_size)
        self.fc = nn.Linear(800, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        #x=self.embedding(x.long())
        out = self.transformer(x)
        out = out.view(out.shape[0],-1)
        out = self.fc(out)
        out = self.fc3(self.fc2(self.fc1(out)))

        out = torch.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_loss = nn.MSELoss()(outputs, targets.view(-1, 1))
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
