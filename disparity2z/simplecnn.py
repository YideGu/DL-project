import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform

class simplecnn(nn.modules.Module):

    def __init__(self):
        super(simplecnn, self).__init__()
        self.conv = nn.Sequential(
        	nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        	nn.ReLU()
        	)
        self.fc = nn.Sequential(
        	nn.Linear(64, 64),
        	nn.Sigmoid(),
        	nn.Linear(64, 1),
        	nn.ReLU()
        	)

    def forward(self, x):
    	x = self.conv(x)
    	x = x.permute(0, 2, 3, 1)
    	x = self.fc(x)
    	return x
