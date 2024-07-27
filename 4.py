import torch
x=torch.rand(3,requires_grad=True)
print(x)
y=x+2
z=y*y*2
print(z)
#z=z.mean()
v=torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
print(v)
z.backward(z)
#print(z)
print(x.grad)