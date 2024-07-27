import numpy as np
import torch 
x=torch.tensor([1,2,3,4,5],dtype=torch.float32)
y=torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)
w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
    return w*x
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')
lr=0.01
n_iters=20
for epoch in range(n_iters):
    y_pred=forward(x)
    l=loss(y, y_pred)
    
    l.backward()
    with torch.no_grad():
        w-=lr*w.grad
    w.grad.zero_()
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')