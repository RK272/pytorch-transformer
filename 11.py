import torch
import torch.nn as nn

import numpy as np
import torch 
x=torch.tensor([[1],[2],[3],[4],[5]],dtype=torch.float32)
y=torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)
#w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
n_samples, n_features=x.shape
print(n_samples, n_features)
input_size=n_features
output_size=n_features
model=nn.Linear(input_size, output_size)
x_test=torch.tensor([5], dtype=torch.float32)

print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')
lr=0.01
n_iters=100

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(n_iters):
    y_pred=model(x)
    l=loss(y, y_pred)
    
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 ==0:
        [w, b]=model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')