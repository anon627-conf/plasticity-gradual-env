import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=47),
            nn.Linear(in_features=84, out_features=47),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    
    
class CReLU(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=self.dim)
    
    
    
class CReLU(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=self.dim)
    


class LayerNormMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=256, num_layers=4):
        super(LayerNormMLP, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.linears.append(nn.Linear(in_d, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x = self.norms[i](x)
            x = torch.relu(x)
        
        out = self.output_layer(x)
        return out
