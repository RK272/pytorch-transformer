import torch
weights=torch.ones(3,requires_grad=True)
for epoch in range(3):
    model_output=(weights*3).sum()
    print(model_output)
    model_output.backward()
    
    print(weights.grad)
    weights.grad.zero_()