'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--reduction', type=str, default='mean', metavar='N',
                    help='Type of reduction to do [choices: sum, mean] (default: mean)')
parser.add_argument('--use-mse', type=bool, default=False, metavar='N',
                    help='Whether to use MSE instead of BCE (default: False)')
parser.add_argument('--convert_path', type=str, default='C:/Program Files/ImageMagick/convert.exe',
                    metavar='N', help='Under windows, specify where convert.exe is located')                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset , batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, embedding_size=2):
        super().__init__()

        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(28*28, 512)
        self.fc1_mu = nn.Linear(512, self.embedding_size) 
        self.fc1_std = nn.Linear(512, self.embedding_size) 

        self.decoder = nn.Sequential( nn.Linear(self.embedding_size, 512), 
                                      nn.ReLU(),
                                      nn.Linear(512, 28*28),
                                      nn.Sigmoid())
    # VAEs sample from a random node z. Backprop cannot flow through a random node.
    # In order to solve this, we randomly sample 'epsilon' from a unit Gaussian,
    # and then simply shift it by the latent distrubtions mean and scale it by its varinace
    # This is called "reparameterization trick".
    # ϵ allows us to reparameterize z in a way that allows backprop to flow through the 
    # deterministic nodes.
    # With this reparameterization, we can now optimize the parameters of the distribution
    # while still maintaining the ability to randomly sample from that distribution.
    # Note: In order to deal with the fact that the network may learn negative values
    # for σ, we'll typically have the network learn log(σ) and exponentiate(exp)) this value 
    # to get the latent distribution's variance.
    def reparamtrization_trick(self, mu, logvar):
        # we divide by two because we are eliminating the negative values
        # and we only care about the absolute possible deviance from standard.
        std = torch.exp(0.5*logvar)
        # epsilon sampled from normal distribution with N(0,1)
        eps = torch.randn_like(std)
        # How to sample from a normal distribution with known mean and variance?
        # https://stats.stackexchange.com/questions/16334/ 
        # (tldr: just add the mu , multiply by the var) . 
        # why we use an epsilon? because without it, backprop wouldnt work.
        return mu + eps*std

    def encode(self, input):
        input = input.view(input.size(0), -1)
        output = F.relu(self.fc1(input))
        # ref: https://www.jeremyjordan.me/variational-autoencoders/
        # Note that we are not using any activation functions here. 
        # in other words our vectors μ and σ are unbounded that is they can take 
        # any values and thus our encoder will be able to learn to generate very 
        # different μ for different classes, clustering them apart, and minimize σ,
        # making sure the encodings themselves don’t vary much for the same sample 
        # (that is, less uncertainty for the decoder). 
        # This allows the decoder to efficiently reconstruct the training data.
        mu = self.fc1_mu(output)
        log_var = self.fc1_std(output)
        z = self.reparamtrization_trick(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
         output = self.decoder(z).view(z.size(0), 1, 28, 28)
         return output
    
    def forward(self, input):
        z, mu, logvar = self.encode(input)
        return self.decode(z), mu, logvar

embedding_size = 2
model = VAE(embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(outputs, inputs, mu, logvar, reduction ='mean', use_mse = False):
    if reduction == 'sum':
        criterion = nn.BCELoss(reduction='sum')
        reconstruction_loss = criterion(outputs, inputs)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + KL
    else:
        if use_mse:
            criterion = nn.MSELoss()
        else: 
            criterion = nn.BCELoss(reduction='mean')
        reconstruction_loss = criterion(outputs, inputs)
        # normalize reconstruction loss
        reconstruction_loss *= 28*28
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        return torch.mean(reconstruction_loss + KL)


def plot_latent_space():
    dataloader_test = DataLoader(test_dataset,
                                 batch_size = len(test_dataset),
                                 num_workers = 2,
                                 pin_memory=True)
    imgs, labels = next(iter(dataloader_test))
    imgs = imgs.to(device)
    z_test,_,_ = model.encode(imgs)
    z_test = z_test.cpu().detach().numpy()

    plt.figure(figsize=(12,10))
    img = plt.scatter(x=z_test[:,0],
                y=z_test[:,1],
                c=labels.numpy(),
                alpha=.4,
                s=3**2,
                cmap='viridis')
    plt.colorbar()
    plt.xlabel('Z[0]')
    plt.ylabel('Z[1]')
    plt.savefig('vae_latent_space.png')
    plt.show()

def display_2d_manifold(model, digit_count=20):
    # display a 2D manifold of the digits
    embeddingsize = model.embedding_size
    # figure with 20x20 digits
    n = digit_count  
    digit_size = 28

    z1 = torch.linspace(-2, 2, n)
    z2 = torch.linspace(-2, 2, n)

    z_grid = np.dstack(np.meshgrid(z1, z2))
    z_grid = torch.from_numpy(z_grid).to(device)
    z_grid = z_grid.reshape(-1, embeddingsize)

    x_pred_grid = model.decode(z_grid)
    x_pred_grid= x_pred_grid.cpu().detach()
    x = make_grid(x_pred_grid, nrow=n).numpy().transpose(1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.imshow(x)
    plt.savefig('vae_digits_2d_manifiold.png')
    plt.show()

def save_animation(model, sample_count = 30, use_mp4=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if os.name == 'nt':
        plt.rcParams["animation.convert_path"] = args.convert_path
    
    z = torch.randn(size = (sample_count, model.embedding_size)).to(device)
    model.eval()

    def animate(i): 
        imgs = model.decode(z * (i * 0.03) + 0.02)
        img_grid = make_grid(imgs).cpu().detach().numpy().transpose(1, 2, 0)
        ax.clear()
        ax.imshow(img_grid)

    anim = animation.FuncAnimation(fig, animate, frames=100, interval=300,
                                   repeat=True, repeat_delay=1000)
    if use_mp4:
        anim.save('vae_off.mp4', writer="ffmpeg", fps=20)
    else:
        anim.save('vae_off.gif', writer="imagemagick", extra_args="convert", fps=20)
    # plt.show()

def train(epoch, reduction='mean', use_mse=False):
    model.train()
    train_loss = 0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recons, mu, logvar = model(imgs)
        loss = loss_function(recons, 
                             imgs,
                             mu, 
                             logvar, 
                             reduction,
                             use_mse)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss = loss/len(imgs) if  (reduction == 'sum') else loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch, reduction='mean', use_mse=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            recons, mu, logvar = model(imgs)
            test_loss += loss_function(recons,
                                       imgs,
                                       mu, 
                                       logvar, 
                                       reduction,
                                       use_mse).item()
            if i == 0:
                n = min(imgs.size(0), 8)
                comparison = torch.cat([imgs[:n], recons[:n]])
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss = test_loss/len(test_loader.dataset) if (reduction == 'sum') else test_loss/len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    if args.reduction =='sum' and args.use_mse:
        print('Warning: reduction=sum will only use BCE. use_mse is ignored!')

    for epoch in range(1, args.epochs + 1):
        train(epoch, args.reduction, args.use_mse)
        test(epoch, args.reduction, args.use_mse)
        with torch.no_grad():
            sample = torch.randn(64, model.embedding_size).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample,'results/sample_' + str(epoch) + '.png')
    save_animation(model, sample_count=30, use_mp4=False)
    plot_latent_space()
    display_2d_manifold(model, digit_count=20)
