import torch
import torch.nn as nn

class NeralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred=torch.sigmoid(out)
        return y_pred
model=NeralNet2(input_size=28*28,hidden_size=5,num_classes=3)
criterion=nn.BCELoss()
