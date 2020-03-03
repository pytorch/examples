from __future__ import print_function
import argparse
import torch
import threading
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--run-mode', type=str, default="multiprocess",
                    help='what mode to (default: multiprocess')
parser.add_argument('--num-threads', type=int, default=2, metavar='N',
                    help='how many training threads to use (default: 2)')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = Net().to(device)
    if args.run_mode == "multiprocess":
        model.share_memory() # gradients are allocated lazily, so they are not shared here
        launcher = mp.Process
        num_workers = args.num_processes
    elif args.run_mode == "multithread":
        # turn model to TorchScript to get rid of GIL
        # model = torch.jit.script(model)
        launcher = threading.Thread
        num_workers = args.num_threads
    else:
        raise RuntimeError(
            "run-mode only support multithread/multiprocess, but got: " + str(args.run_mode)
        )


    workers = []
    # print("num_workers: {}, device: {}".format(num_workers, device))
    for rank in range(num_workers):
        print("before landing thread")
        p = launcher(target=train, args=(rank, args, model, device, dataloader_kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    print("after finish")

    # Once training is complete, we can test the model
    test(args, model, device, dataloader_kwargs)
