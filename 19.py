import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) /np.sum(np.exp(x),axis=0)
x=np.array([2.0,1.0,.01])
output=softmax(x)
print("softmax",output)

x=torch.tensor([2.0, 1.0, .01])
output=torch.softmax(x, dim=0)
print("torch softmax", output)