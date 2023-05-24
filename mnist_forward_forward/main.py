# This code is based on the implementation of Mohammad Pezeshki available at
# https://github.com/mohammadpz/pytorch_forward_forward and licensed under the MIT License.
# Modifications/Improvements to the original code have been made by Vivek V Patel.

import argparse
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam


def get_y_neg(y):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(10))
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)


def overlay_y_on_x(x, y, classes=10):
    x_ = x.clone()
    x_[:, :classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):

        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers = self.layers + [Layer(dims[d], dims[d + 1]).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness = goodness + [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print("training layer: ", i)
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.num_epochs = args.epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(
                1
                + torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % args.log_interval == 0:
                print("Loss: ", loss.item())
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        metavar="LR",
        help="learning rate (default: 0.03)",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no_mps", action="store_true", default=False, help="disables MPS training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="For saving the current Model",
    )
    parser.add_argument(
        "--train_size", type=int, default=50000, help="size of training set"
    )
    parser.add_argument(
        "--threshold", type=float, default=2, help="threshold for training"
    )
    parser.add_argument("--test_size", type=int, default=10000, help="size of test set")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.train_size}
    test_kwargs = {"batch_size": args.test_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )
    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform), **train_kwargs
    )
    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform), **test_kwargs
    )
    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_pos = overlay_y_on_x(x, y)
    y_neg = get_y_neg(y)
    x_neg = overlay_y_on_x(x, y_neg)
    net.train(x_pos, x_neg)
    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    if args.save_model:
        torch.save(net.state_dict(), "mnist_ff.pt")
    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
