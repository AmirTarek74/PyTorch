# Import libraries
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.model_selection import train_test_split
from torchinfo import summary



seeds = 42
torch.manual_seed(seeds)


# Define Model
class NN(nn.Module):
    def __init__(self, inputshape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=inputshape, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=1)
        )


    def forward(self, x):

        return self.model(x)


# Define Data Class
class HouseData(Dataset):
    def __init__(self, file_name):
        super().__init__()
        self.data = pd.read_csv(file_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :].values
        x = transforms.ToTensor()(sample[0:len(sample) - 2].reshape(1, len(sample) - 2)).squeeze(dim=0)
        y = transforms.ToTensor()(sample[-1].reshape(1, 1)).squeeze(dim=0)
        return x, y


# Hyper parameters
batch_size = 16
learning_rate = 0.0001
epochs = 200
input_shape = 79
device = "cuda" if torch.cuda.is_available() else "cpu"
# load Data
data_path = 'FinalTrain2.csv'
data = HouseData(data_path)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=seeds)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

# model, loss & optimizer
model = NN(input_shape).to(device)
criteran = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)
# train
losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    print(f"Epoch {epoch+1} out of {epochs}")
    for idx, batch in enumerate(train_loader):
        x = batch[0].to(device).reshape(batch_size, -1).float()
        y = batch[1].to(device).reshape(batch_size, -1).float()
        output = model(x)
        loss = criteran(output, y)
        train_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        break
    print(f"train loss : {train_loss}")
    losses.append(train_loss/len(train_loader))
area = []
predicted = []
targets = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device).reshape(1, -1).float()
        y = batch[1].to(device).reshape(1, -1).float()
        output = model(x)
        area.append(x[0,3].item())
        predicted.append(output.item())
        targets.append(y.item())

plt.scatter(area, targets, label='Real Price', c='r')
plt.plot(area, predicted, label='Predicted Price')
plt.title('Real Price vs Model output')
plt.legend()
plt.show()