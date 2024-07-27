import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs=4
batch_size=4
learning_rate=0.001

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #normalization

train_dataset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
detaiter=iter(train_loader)
images, labels=next(detaiter)
#imshow(torchvision.utils.make_grid(images))
conv1=nn.Conv2d(3, 6, 5)#3 input channels, 6 output channels, 5x5 kernel
        #self.conv1.weight.data.normal_(0, 0.01)
pool=nn.MaxPool2d(2, 2)
conv2=nn.Conv2d(6, 16, 5)#here 6 is input channelsize because last time ouput was 6
print(images.shape)

x=conv1(images)
print(x.shape)
x=pool(x)
print(x.shape)
x=conv2(x)
print(x.shape)
x=pool(x)
print(x.shape)