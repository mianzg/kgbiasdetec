import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.linear = nn.Linear(input_dim, 1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        return self.linear(x).squeeze()

    def predict(self, x):
        return torch.round(self.sigmoid(self.forward(x)))
