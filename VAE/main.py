from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()

print('Running with CUDA: {0}'.format(cuda))


def print_header(msg):
    print('===>', msg)


assert os.path.exists('data/processed/training.pt'), \
    "Please run python ../mnist/data.py before starting the VAE."

# Data
print_header('Loading data')
with open('data/processed/training.pt', 'rb') as f:
    training_set = torch.load(f)
with open('data/processed/test.pt', 'rb') as f:
    test_set = torch.load(f)

training_data = training_set[0].view(-1, 784).div(255)
test_data = test_set[0].view(-1, 784).div(255)

del training_set
del test_set

# Model
print_header('Building model')


class VAE(nn.Container):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(std.size()), requires_grad=False)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if cuda is True:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


# Training settings
BATCH_SIZE = 150
TEST_BATCH_SIZE = 1000
NUM_EPOCHS = 2

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    batch_data_t = torch.FloatTensor(BATCH_SIZE, 784)
    if cuda:
        batch_data_t = batch_data_t.cuda()
    batch_data = Variable(batch_data_t, requires_grad=False)
    for i in range(0, training_data.size(0), BATCH_SIZE):
        optimizer.zero_grad()
        batch_data.data[:] = training_data[i:i + BATCH_SIZE]
        recon_batch_data, mu, logvar = model(batch_data)
        loss = loss_function(recon_batch_data, batch_data, mu, logvar)
        loss.backward()
        loss = loss.data[0]
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch,
                i + BATCH_SIZE, training_data.size(0),
                float(i + BATCH_SIZE) / training_data.size(0) * 100,
                loss / BATCH_SIZE))


def test(epoch):
    test_loss = 0
    batch_data_t = torch.FloatTensor(TEST_BATCH_SIZE, 784)
    if cuda:
        batch_data_t = batch_data_t.cuda()
    batch_data = Variable(batch_data_t, volatile=True)
    for i in range(0, test_data.size(0), TEST_BATCH_SIZE):
        print('Testing model: {}/{}'.format(i, test_data.size(0)), end='\r')
        batch_data.data[:] = test_data[i:i + TEST_BATCH_SIZE]
        recon_batch_data, mu, logvar = model(batch_data)
        test_loss += loss_function(recon_batch_data, batch_data, mu, logvar)

    test_loss = test_loss.data[0] / test_data.size(0)
    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_loss))


for epoch in range(1, NUM_EPOCHS + 1):
    train(epoch)
    test(epoch)
