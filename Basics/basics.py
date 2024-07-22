import torch
import numpy as np

# initialize tensors

x = torch.eye(3)
x = torch.ones((3, 3))
x = torch.zeros((3, 3))
x = torch.empty((3, 3))

# basic operations
x = torch.rand((3, 3))
y = torch.rand((3, 3))

z = x + y  # torch.add(x,y)
z = x - y  # torch.sub(x,y)
z = x * y  # torch.multiply(x,y)
z = x / y  # torch.div(x,y)

# matrix multiplication
z = torch.matmul(x, y)  # torch.mm(x,y)

# useful functions
z = torch.max(x, dim=0)
z = torch.min(x, dim=1)

# Indexing

batch_size = 32
features = 25
x = torch.rand((batch_size, features))

z = x[0, 0:10]

# Fancy Indexing
x = torch.rand((5, 3))
rows = torch.tensor([1, 4])
cols = torch.tensor([2, 1])
z = x[rows, cols]

# More Advancing Indexing
x = torch.arange(10)

z = x[(x < 2) & (x > 8)]

z = x[x.remainder(2) == 0]

# useful operation
z = torch.where(x > 5, x, x**2)


# Tensor reshaping
x1 = torch.rand((3,5))
x2 = torch.rand((3,5))
z = torch.cat((x1, x2), dim=1)      # dim = 0 for cat under each other, dim = 1 for cat next to each other

batch = 32
x = torch.rand((batch, 2, 5))
print(x.shape)

z = x.permute(0, 2, 1)
print(z.shape)

z = x.unsqueeze(dim = 0)
print(z.shape)

z = z.squeeze(0)
print(z.shape)