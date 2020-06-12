import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(28 * 28, 256), 
            nn.SELU()
        )
        self.layers = nn.Sequential(
            nn.Linear(256, 256), 
            nn.SELU(),
            nn.Linear(256, 256), 
            nn.SELU(),
        )
        self.decode = nn.Sequential(
            nn.Linear(256, 28 * 28), 
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encode(x)
        x = self.layers(x)
        x = self.decode(x)
        x = x.view(1, 28, 28)
        return x