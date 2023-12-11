import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, num_categories):
        super(QFunction, self).__init__()

        # Define the layers for your Q-function
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=1, num_layers=10, batch_first=True)
        self.range_adjusted = 60

    def forward(self, state, action):
        x = torch.cat((state, action), dim=2)
        #print(x.shape)
        lstm_out, _ = self.lstm(x)  # Use unsqueeze to add a sequence dimension
        lstm_out = torch.squeeze(lstm_out)
        #lstm_out = torch.mean(lstm_out,1)
        #print(lstm_out.shape,"^^^^^^^^^^^^^^^^^^^^^^^")
        out = torch.sigmoid(lstm_out) * self.range_adjusted
        #print(out.shape,"^^^^^^^^^^^^^^^^^^^^^^^")
        return out

class ValueFunction(nn.Module):
    def __init__(self, state_dim, num_categories):
        super(ValueFunction, self).__init__()

        # Define the layers for your value function
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=1, num_layers=10, batch_first=True)
        self.range_adjusted = 60

    def forward(self, state):
        lstm_out, _ = self.lstm(state) 
        #print(lstm_out.shape,"^^^^^^^^^^^^^^^^^^^^^^^") # Use unsqueeze to add a sequence dimension
        #lstm_out = torch.mean(lstm_out,1)
        #print(lstm_out.shape,"^^^^^^^^^^^^^^^^^^^^^^^")
        out = torch.sigmoid(lstm_out) * self.range_adjusted
        #print(out.shape,"**********")
        return out

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=2):
        super(PolicyNetwork, self).__init__()

        # Define the layers for your policy network using LSTM
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Output two values (mu and sigma)

    def forward(self, state):
        # Pass the state input through the LSTM layers
        lstm_out, _ = self.lstm(state)
        
        # Extract the last LSTM output
        #print(lstm_out.shape,"^^^^^^^^^^^^^^^^^^^^^^^")
        #print(lstm_out.shape,"@")

        #print(lstm_out.shape,"@")
        #print(lstm_out.shape,"^^^^^^^^^^^^^^^^^^^^^^^")

        # Pass the LSTM output through the fully connected layer
        
        #output = self.fc(lstm_out)
        output=lstm_out
        #print(output.shape,"@")
        mu1 = torch.sigmoid(output[:,:, 0]) * 5
        mu2=  torch.sigmoid(output[:,:, 1]) * 5
        # sigma = torch.sigmoid(output[:,:, 1]) * 2

        return mu1,mu2


