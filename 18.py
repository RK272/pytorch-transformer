import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):
    def __init__(self,transform=None):
        xy=np.loadtxt('./data/wine/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x=xy[:,1:]
        self.y=xy[:,[0]]
        self.n_sample=xy.shape[0]
        self.transform=transform

    def __getitem__(self, index):
        sample=self.x[index], self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample
    def __len__(self):
        return self.n_sample
    
class ToTensor:
    def __call__(self,sample):
        inputs,targets=sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)

class multransform:
    def __init__(self,factor):
        self.factor=factor
    def __call__(self, sample):
        inputs,targets=sample
        inputs*=self.factor
        return inputs,targets
    
dataset=WineDataset(transform=ToTensor())
first_data=dataset[0]
features,labels=first_data
print(features,labels)
print(type(features), type(labels))

dataset1=WineDataset(transform=None)
first_data1=dataset1[0]
features1,labels1=first_data1
print(features1,labels1)
print(type(features1), type(labels1))

composed=torchvision.transforms.Compose([ToTensor(), multransform(2)])
dataset2=WineDataset(transform=composed)
first_data2=dataset2[0] 

features2, labels2=first_data2
print(features2, labels2)
print(type(features2), type(labels2))

