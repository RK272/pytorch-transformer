import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
import math
"""
epoch=1 forward and backward pass of all training samples
batch_size=number of training samples in one forward and backward pass
number of iterations=number of passes, each pass using [batch_size] number of samples
eg 100 samplae, batchsize=20, 100/20=5 iteration for 1 epoch


"""
class WineDataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('./data/wine/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]])
        self.n_sample=xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample
    
dataset=WineDataset()
first_data=dataset[0]
features,labels=first_data
print(features,labels)

