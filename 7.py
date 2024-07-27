import torch
x=torch.tensor(1)
y=torch.tensor(2)
w=torch.tensor(1.0,requires_grad=True)
y_hat=w*x
loss=(y-y_hat)**2
print(loss)
loss.backward()
print(w.grad)   