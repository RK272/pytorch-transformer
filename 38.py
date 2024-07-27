import torch
import torch.nn as nn
#save on gpu load on cpu
device=torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), 'model_weights.pth')

device=torch.device("cpu")
model=Model(*args,**kwargs)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))

#save on gpu load on gpu
device=torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), 'model_weights.pth')


model=Model(*args,**kwargs)
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)

#save on cpu load on gpu
torch.save(model.state_dict(), 'model_weights.pth')

device=torch.device("cuda")
model=Model(*args,**kwargs)
model.load_state_dict(torch.load('model_weights.pth', map_location="cuda:0"))
model.to(device)

