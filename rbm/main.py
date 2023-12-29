import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 200
epochs = 10
lr = 0.001
nh = 1000
data = datasets.MNIST(
            root='~/.pytorch/MNIST_data/',
            download=True
        ).data
data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)

train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

def show_imgs(img_tensors):
  fig, axs = plt.subplots(nrows=10, ncols=10)
  for i in range(100):
    axs[i//10, i%10].imshow(img_tensors[i].view(28, 28), cmap='binary_r')
    axs[i//10, i%10].set_axis_off()
  plt.show()




class RBM(nn.Module):
    def __init__(self, n_v, n_h):
        super().__init__()
        self.nv = n_v
        self.nh = n_h
        self.b = nn.Parameter(torch.normal(0, 1, size=[n_v], dtype=torch.float32, requires_grad=True))
        self.c = nn.Parameter(torch.normal(0, 1, size=[n_h], dtype=torch.float32, requires_grad=True))
        self.W = nn.Parameter(torch.normal(0, 1, size=(n_v, n_h), dtype=torch.float32, requires_grad=True))
        self.sig = nn.Sigmoid()

    def forward(self, v_d, v_m):
        with torch.no_grad():
            h_d = self.h_given_v(v_d)
            v_m, h_m = self.gibbs_update(v_m, 50)
        positive_phase = self.b @ v_d.T + self.c @ h_d.T + ((v_d @ self.W) * h_d).sum(dim=-1)
        negative_phase = self.b @ v_m.T + self.c @ h_m.T + ((v_m @ self.W) * h_m).sum(dim=-1)
        llh = positive_phase - negative_phase
        m = llh.size(0)
        llh = -(llh.sum())/m
        return llh, v_m

    def gibbs_update(self, v, k):
        for i in range(k):
            h = self.h_given_v(v)
            v = self.v_given_h(h)
        return v, h

    def h_given_v(self, v):
        pmf = self.sig(self.c + v @ self.W)
        return torch.bernoulli(pmf)

    def v_given_h(self, h):
        pmf = self.sig(self.b +  h @ self.W.T)
        return torch.bernoulli(pmf)

    def sample(self, n):
        with torch.no_grad():
            v = torch.bernoulli(torch.rand(n, self.nv)).to(self.W.device)
            v, _ = self.gibbs_update(v, 100)
            return v

    

model = RBM(784, nh).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
for i in range(epochs):
  for batch in train_loader:
    v = torch.bernoulli(torch.rand(batch_size,784)).to(device)
    b = batch.view(batch_size, -1).to(device)
    loss, _ = model(b, v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if i % 10 == 0:
    print(f"step: {i}/{epochs} loss: {loss.item()}")

mse = nn.MSELoss()
def reconst_err(model):
  with torch.no_grad():
    err = 0.0
    for batch in train_loader:
      b = batch.view(batch_size, -1).to(device)
      v = model(b.to(device), b.to(device))[1]
      err += mse(b,v)
    return err/len(train_loader)


show_imgs(model.sample(100).cpu())