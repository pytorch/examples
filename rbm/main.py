import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

# the restricted boltzmann machine model
class RBM(nn.Module):
    def __init__(self, n_v, n_h, k):
        super().__init__()
        self.nv = n_v  #the number of visible units
        self.nh = n_h  #the number of hidden units
        #weights and biases
        self.b = nn.Parameter(torch.normal(0, 1, size=[n_v], dtype=torch.float32, requires_grad=True))  
        self.c = nn.Parameter(torch.normal(0, 1, size=[n_h], dtype=torch.float32, requires_grad=True))
        self.W = nn.Parameter(torch.normal(0, 1, size=(n_v, n_h), dtype=torch.float32, requires_grad=True))
        self.sig = nn.Sigmoid()
        self.k = k  #number of gibbs steps

    def forward(self, v_d, v_m):
        with torch.no_grad():
            h_d = self.h_given_v(v_d)    #the calculation of hidden units of the data from the visible units
            v_m, h_m = self.gibbs_update(v_m, self.k)   # markov chain to get a fair sample from the model
        positive_phase = self.b @ v_d.T + self.c @ h_d.T + ((v_d @ self.W) * h_d).sum(dim=-1)   #the positive phase of the loglikelihood
        negative_phase = self.b @ v_m.T + self.c @ h_m.T + ((v_m @ self.W) * h_m).sum(dim=-1)   #the negative phase of the loglikelihood
        llh = positive_phase - negative_phase     
        m = llh.size(0)     #number of samples
        llh = -(llh.sum())/m
        return llh, v_m    # return the loss and the visible units sampled from the model in case you want to train with PCD

    def gibbs_update(self, v, k):  #this method is for the markov chain mixing
        for _ in range(k):
            h = self.h_given_v(v)
            v = self.v_given_h(h)
        return v, h

    def h_given_v(self, v):    #this return a sample from the conditional  probability distibution p(h|v)
        pmf = self.sig(self.c + v @ self.W)
        return torch.bernoulli(pmf)

    def v_given_h(self, h):    #this return a sample the conditional  probability distibution p(v|h)
        pmf = self.sig(self.b +  h @ self.W.T)
        return torch.bernoulli(pmf)

    def sample(self, n):   #this samples from the model starting from random intialisation and mixing for 100 step
        with torch.no_grad():
            v = torch.bernoulli(torch.rand(n, self.nv)).to(self.W.device)
            v, _ = self.gibbs_update(v, 100)
            return v

    # the train function implement the naive approach of intializing from a random distribution every gradient step 
    # it can be easily modifed to be CD by removing v and passing b to the model
    # also PCD can be accomplished by moving v out of the loop and renaming the underscore to v 

def train(model, train_loader, optimizer,epochs, batch_size, device):   
  for i in range(epochs + 1):
    for batch in train_loader:
      v = torch.bernoulli(torch.rand(batch_size,784)).to(device)
      b = batch.view(batch_size, -1).to(device)
      loss, _ = model(b, v)  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    if i % 10 == 0:
      print(f"step: {i}/{epochs} loss: {loss.item()}")


  #the best way to evaluate the model is by looking to the samples created by the model in "model_samples.png"
    #but this evaluation is based on the reconstraction error so it can give you a false idea about its capacity
def evaluate(model, train_loader, mse, batch_size, device):
  with torch.no_grad():
    err = 0.0
    for batch in train_loader:
      b = batch.view(batch_size, -1).to(device)
      v = model(b.to(device), b.to(device))[1]
      err += mse(b,v)
    print(f"Reconstraction error: {err/len(train_loader)}") 


#this method takes image tensors and save them as png
def show_imgs(img_tensors):
  fig, axs = plt.subplots(nrows=10, ncols=10)
  for i in range(100):
    axs[i//10, i%10].imshow(img_tensors[i].view(28, 28), cmap='binary_r')
    axs[i//10, i%10].set_axis_off()
  plt.savefig('model_samples.png')
  plt.show()


def main():
  """Parses the arguments for training a RBM."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--batch", type=int, default=200, help="Batch size for training.")
  parser.add_argument(
      "--epochs", type=int, default=10, help="Number of epochs for training.")
  parser.add_argument(
      "--lr", type=float, default=0.001, help="Learning rate for training.")
  parser.add_argument(
      "--nh", type=int, default=1000, help="Number of hidden units for the RBM.")
  parser.add_argument(
      "--k", type=int, default=50, help="Number of Gibbs steps during training")
  parser.add_argument(
     '--save', help='Save the model after training', action='store_true')

  
  args = parser.parse_args()

  batch_size = args.batch
  epochs = args.epochs
  lr = args.lr
  nh = args.nh
  k = args.k

  device = "cuda" if torch.cuda.is_available() else "cpu"


#the data used is mnist
  data = datasets.MNIST(
              root='~/.pytorch/MNIST_data/',
              download=True
          ).data
  data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)  #here i transform it into binary form just 0 and 1 pixels because the model has binary units

  print(f"Training device: {device}")

  train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

  model = RBM(784, nh, k).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  mse = nn.MSELoss()

  train(model, train_loader, optimizer,epochs, batch_size, device)
  evaluate(model, train_loader, mse, batch_size, device)

  if args.save:
      torch.save(model.state_dict(), "rbm.pt")



  show_imgs(model.sample(100).cpu())


if __name__ == "__main__":
   main()
