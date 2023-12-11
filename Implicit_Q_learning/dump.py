import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, num_categories):
        super(QFunction, self).__init__()

        # Define the layers for your Q-function
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.range_adjusted = 60

    def forward(self, state, action):
        # Concatenate the state and action inputs
        x = torch.cat((state, action), dim=1)
        
        # Pass the concatenated input through the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))*self.range_adjusted

        return out




class ValueFunction(nn.Module):
    def __init__(self, state_dim, num_categories):
        super(ValueFunction, self).__init__()

        # Define the layers for your value function
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.range_adjusted = 60

    def forward(self, state):
        # Pass the state input through the layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))*self.range_adjusted

        return out




class PolicyNetwork(nn.Module):
    def __init__(self, state_dim):
        super(PolicyNetwork, self).__init__()

        # Define the layers for your policy network
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output a single scalar value

        # Define the output range

    def forward(self, state , action):
        # Pass the state input through the layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = torch.tanh(self.fc3(x)) 
        mu= torch.sigmoid(output[:,0])*10
        sigma = torch.sigmoid(output[:,1])*10
        normal_dist = Normal(mu, sigma)
        
        log_prob = normal_dist.log_prob(action)
        log_prob = torch.diagonal(log_prob)
        return log_prob

