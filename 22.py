import torch
import torch.nn as nn
import numpy as np

loss=nn.CrossEntropyLoss()
#3 samples
y=torch.tensor([2, 0, 1])
#y=torch.tensor([0])
#nsamples*nclasses=3*3
y_pred_good=torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[2.0,1.0,0.1]])
y_pred_bad=torch.tensor([[2.5, 2.0, 0.3],[0.5, 2.0, 0.3],[0.5, 2.0, 0.3]])
l1=loss(y_pred_good, y)
l2=loss(y_pred_bad, y)
print(f'Loss1: {l1.item():.4f}')
print(f'Loss2: {l2.item():.4f}')
_, prediction1=torch.max(y_pred_good, dim=1)
_, prediction2=torch.max(y_pred_bad, dim=1)
print(f'Prediction1: {prediction1.item()}')
print(f'Prediction2: {prediction2.item()}')