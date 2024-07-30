# Import libraries
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.model_selection import train_test_split

seeds = 42
torch.manual_seed(seeds)


# Define Model
class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(3, 3), stride=1,
                               padding=1)  #same convolution
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1,
                               padding=1)  # same convolution
        self.fc1 = nn.Linear(7 * 7 * 16, num_classes)

    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU(inplace=True)(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Define Data Class

# Hyper parameters
batch_size = 16
learning_rate = 0.0001
epochs = 10
input_channels = 1
num_classes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
# load Data
data = datasets.MNIST(root='MINST_DATA/', train=True, download=True, transform=transforms.ToTensor())

train_data, test_data = train_test_split(data, test_size=0.2, random_state=seeds)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# model, loss & optimizer
model = CNN(input_channels, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)
# train
for epoch in range(epochs):
    model.train()
    num_samples = 0
    correct_smaples = 0
    print(f"Epoch {epoch + 1} out of {epochs}")
    for idx, batch in enumerate(train_loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        output = model(x)
        loss = criterion(output, y)
        _, predication = output.max(1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        correct_smaples += (predication == y).sum().item()
        num_samples += x.shape[0]
    print(f"Accuracy for epoc {epoch + 1} out of {epochs} = {correct_smaples / num_samples * 100 : .2f}%")

num_samples = 0
correct_smaples = 0
for batch in test_loader:
    x = batch[0].to(device)
    target = batch[1].to(device)
    out = model(x)
    _, predication = out.max(1)
    correct_smaples += (predication == target).sum().item()
    num_samples += x.shape[0]
print(f"Test Accuracy = {correct_smaples/num_samples *100 : .2f}%")
