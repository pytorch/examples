import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


def train(rank, args, model, device, dataloader_kwargs):
    print("thread started")
    torch.manual_seed(args.seed + rank)

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    begin_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_begin_time = time.time()
        train_epoch(rank, epoch, args, model, device, train_loader, optimizer)
        epoch_end_time = time.time()
        print('Rank {}\tEpoch: {} {} images/sec'.format(
            rank, epoch, int(len(train_dataset) / (epoch_end_time - epoch_begin_time))))

    end_time = time.time()
    # print("Total training time: {}s".format(end_time - begin_time))

def test(args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    test_epoch(model, device, test_loader)


def train_epoch(rank, epoch, args, model, device, data_loader, optimizer):
    model.train()
    print("beginning epoch: {}".format(epoch))
    begin_epoch_time = time.time()
    backward_time = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        begin_backward_time = time.time()
        loss.backward()
        end_backward_time = time.time()
        backward_time += end_backward_time - begin_backward_time
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
            # print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     rank, epoch, batch_idx * len(data), len(data_loader.dataset),
            #     100. * batch_idx / len(data_loader), loss.item()))

    end_epoch_time = time.time()
    print("ending epoch: {} with time spent: {}, and backward spent: {}".format(epoch, end_epoch_time - begin_epoch_time, backward_time))

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
