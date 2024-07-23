import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self, dim, batch_size, learning_rate=0.0001):
        super(LogisticRegression, self).__init__()
        self.w = torch.randn(dim, 1)
        self.b = torch.randn(1, 1)
        self.batch_size = batch_size
        self.grads = {
            "dw": torch.rand(dim, 1),
            "db": torch.rand(1, 1)
        }
        self.lr = learning_rate


    def criterion(self, yhat, y):
        m = y.size()[1]
        error = -(1 / m) * torch.sum(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
        return error

    def optmizer(self, x, yhat, y):
        self.grads["dw"] = (1 / x.shape[1]) * torch.mm(x.T, (yhat - y))
        self.grads["db"] = ((1 / x.shape[1]) * torch.sum(yhat - y)).reshape(1, 1)
        self.w -= self.grads["dw"] * self.lr
        self.b -= self.grads["db"] * self.lr

    def train(self, x, y):
        z = torch.mm(x, self.w) + self.b
        yhat = 1 / (1 + torch.exp(-z))
        self.optmizer(x, yhat, y)
    def predict(self, x):
        z = torch.mm(x, self.w) + self.b
        yhat = 1 / (1 + torch.exp(-z))
        return torch.round(yhat)



seeds = 42
torch.manual_seed(seeds)

# Hyper Parameters
lr = 0.001
Epochs = 100
batch_size = 32

# Data
data = datasets.MNIST(root='MINST_DATA/', train=True, download=True, transform=transforms.ToTensor())

modified_data = []
for example in data:
    if example[1]==0:
        label = 0
    else:
        label = 1
    modified_data.append((example[0], label))
train_data, test_data = train_test_split(modified_data, test_size=0.2, random_state=seeds)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Train
input_shape = 28 * 28
model = LogisticRegression(input_shape,batch_size, lr)

for epoch in range(Epochs):
    num_samples = 0
    correct_smaples = 0
    for example in train_loader:
        img = example[0].reshape(batch_size, -1)
        target = example[1].unsqueeze(dim=1)
        model.train(img, target)
        predication = model.predict(img)
        correct_smaples += (predication==target).sum().item()
        num_samples += img.shape[0]
    print(f"Accuracy for epoc {epoch+1} out of {Epochs} = {correct_smaples/num_samples *100 : .2f}%")

num_samples = 0
correct_smaples = 0
for batch in test_loader:
    x = batch[0].reshape(batch_size, -1)
    target = batch[1].unsqueeze(dim=1)
    predication = model.predict(img)
    correct_smaples += (predication == target).sum().item()
    num_samples += img.shape[0]
print(f"Test Accuracy = {correct_smaples/num_samples *100 : .2f}%")







