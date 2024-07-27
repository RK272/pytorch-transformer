import torch
x=torch.rand(3,requires_grad=True)
print(x)
y=x+2
z=y*y*2
z=z.mean()
z.backward()
print(z)
print(x.grad)
