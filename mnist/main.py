from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

cuda = torch.cuda.is_available()

def print_header(msg):
    print('===>', msg)

if not os.path.exists('data/processed/training.pt'):
    import data

# Data
print_header('Loading data')
with open('data/processed/training.pt', 'rb') as f:
    training_set = torch.load(f)
with open('data/processed/test.pt', 'rb') as f:
    test_set = torch.load(f)

training_data = training_set[0].view(-1, 1, 28, 28).div(255)
training_labels = training_set[1]
test_data = test_set[0].view(-1, 1, 28, 28).div(255)
test_labels = test_set[1]

del training_set
del test_set

# Model
print_header('Building model')
class Net(nn.Container):
    def __init__(self):
        super(Net, self).__init__(
            conv1 = nn.Conv2d(1, 20, 5),
            pool1 = nn.MaxPool2d(2, 2),
            conv2 = nn.Conv2d(20, 50, 5),
            pool2 = nn.MaxPool2d(2, 2),
            fc1   = nn.Linear(800, 500),
            fc2   = nn.Linear(500, 10),
            relu  = nn.ReLU(),
            softmax = nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 800)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(x)

model = Net()
if cuda == True:
    model.cuda()

criterion = nn.NLLLoss()

# Training settings
BATCH_SIZE = 150
TEST_BATCH_SIZE = 1000
NUM_EPOCHS = 2

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

def train(epoch):
    batch_data_t = torch.FloatTensor(BATCH_SIZE, 1, 28, 28)
    batch_targets_t = torch.LongTensor(BATCH_SIZE)
    if cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, requires_grad=False)
    batch_targets = Variable(batch_targets_t, requires_grad=False)
    for i in range(0, training_data.size(0), BATCH_SIZE):
        optimizer.zero_grad()
        batch_data.data[:] = training_data[i:i+BATCH_SIZE]
        batch_targets.data[:] = training_labels[i:i+BATCH_SIZE]
        loss = criterion(model(batch_data), batch_targets)
        loss.backward()
        loss = loss.data[0]
        optimizer.step()
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch,
            i+BATCH_SIZE, training_data.size(0),
            float(i+BATCH_SIZE)/training_data.size(0)*100, loss))

def test(epoch):
    test_loss = 0
    batch_data_t = torch.FloatTensor(TEST_BATCH_SIZE, 1, 28, 28)
    batch_targets_t = torch.LongTensor(TEST_BATCH_SIZE)
    if cuda:
        batch_data_t = batch_data_t.cuda()
        batch_targets_t = batch_targets_t.cuda()
    batch_data = Variable(batch_data_t, volatile=True)
    batch_targets = Variable(batch_targets_t, volatile=True)
    correct = 0
    for i in range(0, test_data.size(0), TEST_BATCH_SIZE):
        print('Testing model: {}/{}'.format(i, test_data.size(0)), end='\r')
        batch_data.data[:] = test_data[i:i+TEST_BATCH_SIZE]
        batch_targets.data[:] = test_labels[i:i+TEST_BATCH_SIZE]
        output = model(batch_data)
        test_loss += criterion(output, batch_targets)
        pred = output.data.max(1)[1]
        correct += pred.long().eq(batch_targets.data.long()).cpu().sum()

    test_loss = test_loss.data[0]
    test_loss /= (test_data.size(0) / TEST_BATCH_SIZE) # criterion averages over batch size
    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, test_data.size(0),
        float(correct)/test_data.size(0)*100))

for epoch in range(1, NUM_EPOCHS+1):
    train(epoch)
    test(epoch)
