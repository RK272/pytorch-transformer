import torch
import torch.nn as nn

class  Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model=Model(input_size=6)
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

print(optimizer,model.state_dict())
checkpoint={
    'epoch':90,
    'model_state':model.state_dict(),
    'optimizer_state':optimizer.state_dict()
}
torch.save(checkpoint, 'checkpoint.pth')

loaded_checkpoint=torch.load('checkpoint.pth')
epoch=loaded_checkpoint['epoch']
model=model=Model(input_size=6)
optimizer=torch.optim.SGD(model.parameters(),lr=0)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])

print(optimizer.state_dict())