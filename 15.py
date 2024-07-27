import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Implement a custom Dataset
class WineDataset(Dataset):
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # Here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    def __getitem__(self, index):
        # Support indexing such that dataset[i] can be used to get i-th sample
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # We can call len(dataset) to return the size
        return self.n_samples

# Ensure main guard for multiprocessing
if __name__ == '__main__':
    # Create dataset
    dataset = WineDataset()

    # Load dataset with DataLoader
    train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    # Convert to an iterator and look at one random sample
    dataiter = iter(train_loader)
    data = next(dataiter)
    features, labels = data
    print(features, labels)

    # Dummy Training loop
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
            # Run your training process
            if (i + 1) % 5 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations} | Inputs {inputs.shape} | Labels {labels.shape}')
