from __future__ import print_function
import platform
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# some constants
NUM_HIDDEN = 100

class FizzBuzzModel(nn.Module):
    def __init__(self, D_in, D_out, hidden):
        super(FizzBuzzModel,self).__init__()
        self.inputLayer = nn.Linear(D_in, hidden)
        self.relu = nn.ReLU()
        self.outputLayer = nn.Linear(hidden, D_out)
        
    def forward(self,x):
        x = self.inputLayer(x)
        x = self.relu(x)
        out = self.outputLayer(x)
        return out

# encode num to binary format array
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])[::-1]

# get fizz/buzz value
def fizzbuzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    # enter train mode
    model.train()

    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target) # calc loss

        loss.backward()
        optimizer.step()

        total += target.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:0>4d} [{:0>4d}/{:0>4d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                total,
                # batch_idx * len(data),
                len(train_loader.dataset),
                (total + (epoch - 1) * len(train_loader.dataset)) * 100 / (len(train_loader.dataset) * args.epochs),
                loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    # enter test mode
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--num-digits', type=int, default=12, metavar='N',
                        help='traing digits number (default: 12)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 0 if platform.system() == 'Windows' else 4,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    # prepare training / testing data
    x = torch.tensor([binary_encode(i, args.num_digits) for i in range(101, 2 ** args.num_digits)], dtype=torch.float)
    y = torch.tensor([fizzbuzz_encode(i) for i in range(101, 2 ** args.num_digits)], dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset=dataset, **kwargs)

    x_t = torch.tensor([binary_encode(i, args.num_digits) for i in range(1, 101)], dtype=torch.float)
    y_t = torch.tensor([fizzbuzz_encode(i) for i in range(1, 101)], dtype=torch.long)

    dataset_t = TensorDataset(x_t, y_t)
    loader_t = DataLoader(dataset_t, batch_size=args.test_batch_size)

    # init the model
    model = FizzBuzzModel(args.num_digits, 4, NUM_HIDDEN).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training & training
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loader, optimizer, criterion, epoch)

        # testing the train result
        test(model, device, loader_t)

    if args.save_model:
        torch.save(model.state_dict(), "fizzbuzz_nn.pt")


if __name__ == '__main__':
    main()
