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
for params in model.parameters():
    print(params)
FILE="model.pth"
FILE1="model1.pth"
#torch.save(model,FILE)
loaded_model=Model(input_size=6)
loaded_model.load_state_dict(torch.load(FILE1))
loaded_model.eval()
for params in loaded_model.parameters():
    print(params)
#torch.save(model.state_dict(),FILE1)
print(model.state_dict())