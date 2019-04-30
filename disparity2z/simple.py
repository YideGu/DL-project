import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class simple(nn.modules.Module):

    def __init__(self):
        super(simple, self).__init__()
        self.fc = nn.Sequential(
        	nn.Linear(1, 100),
        	nn.Sigmoid(),
        	nn.Linear(100, 1),
        	nn.ReLU()
        	)

    def forward(self, x):
    	return self.fc(x)