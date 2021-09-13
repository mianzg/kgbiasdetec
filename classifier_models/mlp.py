import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, dimensions, dropout=0, activation=nn.ReLU, device=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.binary = (dimensions[-1] == 1)
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(dimensions[:-2], dimensions[1:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), activation()
                )
            )
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        self.layers.append(
            nn.Linear(dimensions[-2], dimensions[-1])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.binary:
            x = x.squeeze(1)
        return x

    def predict(self, x):
        if self.binary:
            return torch.round(torch.sigmoid(self.forward(x)))
        return self.forward(x)