import torch
import torch.nn as nn

class  Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model=Model(input_size=6)
FILE="model.pth"
torch.save(model,FILE)
