import torch
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, input_Shape, output_shape, learningRate=0.0001):
        super(LinearRegression,self).__init__()
        self.w = torch.rand((output_shape[1], input_Shape[1]), requires_grad=True)
        self.b = torch.rand((1, output_shape[1]), requires_grad=True)
        self.preds = torch.rand(output_shape)
        self.lr = learningRate

    def criterion(self,t1,t2):
        diff = t1-t2
        return torch.sum(diff*diff)/diff.numel()

    def step(self):
        with torch.no_grad():
            self.w -= self.w.grad * self.lr
            self.b -= self.b.grad * self.lr
            self.w.grad.zero_()
            self.b.grad.zero_()

    def train(self, x, y):
        self.preds = torch.mm(x, self.w.t()) + self.b
        loss = self.criterion(self.preds, y)
        loss.backward()
        self.step()


# Hyper Parameters
lr = 0.01
Epochs = 1000


# Data
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float).reshape(10, 1)
y = (2 * x + 3)
print(y.shape)
input_shape = x.shape
output_shape = y.shape
# Train
model = LinearRegression(input_shape, output_shape, lr)

for epoch in range(Epochs):
    model.train(x, y)

predictions = x * model.w.item() + model.b.item()
print(f" w =  {model.w}, b = {model.b}")
plt.scatter(x.numpy().ravel(), y.numpy().ravel(), label='Targets', c='g')
plt.plot(x.numpy().ravel(), predictions.numpy().ravel(), label='Predictions')
plt.legend()
plt.show()

