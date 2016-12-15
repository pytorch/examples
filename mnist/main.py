from __future__ import print_function
import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batchSize', type=int, default=64, metavar='input batch size')
parser.add_argument('--testBatchSize', type=int, default=1000, metavar='input batch size for testing')
parser.add_argument('--trainSize', type=int, default=1000, metavar='Train dataset size (max=60000). Default: 1000')
parser.add_argument('--nEpochs', type=int, default=2, metavar='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='Learning Rate. Default=0.01')
parser.add_argument('--momentum', type=float, default=0.5, metavar='Default=0.5')
parser.add_argument('--seed', type=int, default=123, metavar='Random Seed to use. Default=123')
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
if cuda == True:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('data/processed/training.pt'):
    import data

# Data
print('===> Loading data')
with open('data/processed/training.pt', 'rb') as f:
    training_set = torch.load(f)
with open('data/processed/test.pt', 'rb') as f:
    test_set = torch.load(f)

training_data = training_set[0].view(-1, 1, 28, 28).div(255)
training_data = training_data[:opt.trainSize]
training_labels = training_set[1]
test_data = test_set[0].view(-1, 1, 28, 28).div(255)
test_labels = test_set[1]

del training_set
del test_set

print('===> Building model')
class Net(nn.Container):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)
        self.relu  = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(x)

model = Net()
if cuda == True:
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

def train(epoch):
    # create buffers for mini-batch
    batch_data = torch.FloatTensor(opt.batchSize, 1, 28, 28)
    batch_targets = torch.LongTensor(opt.batchSize)
    if cuda:
        batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

    # create autograd Variables over these buffers
    batch_data, batch_targets = Variable(batch_data), Variable(batch_targets)

    for i in range(0, training_data.size(0)-opt.batchSize+1, opt.batchSize):
        start, end = i, i+opt.batchSize
        optimizer.zero_grad()
        batch_data.data[:] = training_data[start:end]
        batch_targets.data[:] = training_labels[start:end]
        output = model(batch_data)
        loss = criterion(output, batch_targets)
        loss.backward()
        loss = loss.data[0]
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'
              .format(epoch, end, opt.trainSize, float(end)/opt.trainSize*100, loss))

def test(epoch):
    # create buffers for mini-batch
    batch_data = torch.FloatTensor(opt.testBatchSize, 1, 28, 28)
    batch_targets = torch.LongTensor(opt.testBatchSize)
    if cuda:
        batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

    # create autograd Variables over these buffers
    batch_data = Variable(batch_data, volatile=True)
    batch_targets = Variable(batch_targets, volatile=True)

    test_loss = 0
    correct = 0

    for i in range(0, test_data.size(0), opt.testBatchSize):
        batch_data.data[:] = test_data[i:i+opt.testBatchSize]
        batch_targets.data[:] = test_labels[i:i+opt.testBatchSize]
        output = model(batch_data)
        test_loss += criterion(output, batch_targets)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.long().eq(batch_targets.data.long()).cpu().sum()

    test_loss = test_loss.data[0]
    test_loss /= (test_data.size(0) / opt.testBatchSize) # criterion averages over batch size
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_data.size(0),
        float(correct)/test_data.size(0)*100))

for epoch in range(1, opt.nEpochs+1):
    train(epoch)
    test(epoch)
