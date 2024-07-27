import torch
import torch.nn as nn
import torch.nn.functional as F

class NeralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = self.sigmoid(out)
        return y_pred
    
class NeuralNet(nn.module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        #self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,1)

    def forward(self, x):
        out=torch.relu(self.linear1(x))
        #out=self.relu(out)
        out=torch.sigmoid(self.linear2(out))
        return out

class NeuralNet(nn.module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        #self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,1)

    def forward(self, x):
        out=F.relu(self.linear1(x))
        #out=self.relu(out)
        out=torch.sigmoid(self.linear2(out))
        return out