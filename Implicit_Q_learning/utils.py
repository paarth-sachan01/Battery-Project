import torch
import torch.nn as nn

class ExpectileLoss(nn.Module):
    def __init__(self, tau=0.7):
        super(ExpectileLoss, self).__init__()
        self.tau = tau

    def forward(self, predictions, targets):
        residuals = predictions - targets
        loss = torch.abs(self.tau - (residuals < 0).long()) * residuals**2
        return loss.mean()
        
class Exponential_loss(nn.Module):
    def __init__(self, beta=2e-4):
        super(Exponential_loss, self).__init__()
        self.beta =beta

    def forward(self, q_values, state_values):
        residuals = (q_values - state_values)+1
        loss = torch.exp(self.beta*(residuals))
        return loss.mean()


