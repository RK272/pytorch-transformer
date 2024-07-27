import torch 
import torch.nn as nn
import torchvision
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('runs/mnist2')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hyperparama
input_size=784#28*28
hidden_size=100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.01
#MNIST
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
example=iter(train_loader)
samples,labels=next(example)
print(samples.shape,labels.shape)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)#no softmax because for multiclass we applied loss as crossentropy it already aplly softmax
        return out
model=NeuralNet(input_size, hidden_size, num_classes)
criterion=nn.CrossEntropyLoss()#it will apply softmax
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()
#sys.exit()
n_total_steps=len(train_loader)
running_loss=0.0
running_correct=0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        #forward
        outputs=model(images)
        loss=criterion(outputs, labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _, predicted=torch.max(outputs, 1)
        running_correct+=(predicted==labels).sum().item()
        if (i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epochs}, step{i+1}/{n_total_steps}, loss={loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps+i) 
            writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps+i)
            running_loss=0.0
            running_correct=0.0
labels=[]
preds=[]
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images, labels1 in test_loader:
        images=images.reshape(-1, 28*28).to(device)
        labels1=labels1.to(device)
        outputs=model(images)
        _, predicted=torch.max(outputs, 1)
        n_samples+=labels1.size(0)
        n_correct+=(predicted==labels1).sum().item()
        class_prediction=[F.softmax(output,dim=0) for output in outputs]
        preds.append(class_prediction)
        labels.append(predicted)
    preds=torch.cat([torch.stack(batch) for batch in preds])
    labels=torch.cat(labels)
    acc=100.0*n_correct/n_samples
    print(f'accuracy={acc}')

    classes =range(10)
    for i in classes:
        labels_i=labels==i
        preds_i=preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, 0)
        writer.close()