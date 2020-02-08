import time
import torch
import torch.distributed.rpc as rpc
import os
from threading import Lock
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# --------- MNIST Network to train, from pytorch/examples --------------------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Syncrhnous RPC to run a method remotely and get a result. The method should be a class method corresponding to
# Given an RRef, return the result of calling the passed in method on the value held by the RRef. This call is done on the remote node that owns the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        model = Net()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return out

    # Use dist autograds to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        return grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes paramters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


param_server = None
global_lock = Lock()
# Ensure that we get only one handle to the ParameterServer.


def get_parameter_server():
    global param_server
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer()
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    rpc.shutdown()


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = Net()
        self.model = model
        # TODO take this in as an arg
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=())

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, x, cid):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output


def run_training_loop(rank, train_loader, test_loader):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet()
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            print("Training batch {}".format(i))
            model_output = net(data, cid)
            loss = F.nll_loss(model_output, target)
            print(loss)
            dist_autograd.backward([loss])
            param_rrefs = net.get_global_param_rrefs()
            opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
            opt.step()
            # verify that we have remote gradients
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}

    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, net)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data, -1)
            pred = out.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct
    print("Accuracy {}".format(correct_sum / len(test_loader.dataset)))


# Main loop for trainers.
def run_worker(rank, world_size, train_loader, test_loader):
    rpc.init_rpc(
        name="trainer_{}".format(rank),
        rank=rank,
        world_size=world_size)

    run_training_loop(rank, train_loader, test_loader)
    rpc.shutdown()

# --------- Launcher --------------------


if __name__ == '__main__':
    start = time.time()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Get data to train on
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=32, shuffle=True, )
    processes = []
    # Run num_trainers workers, plus 1 for the parameter serever.
    num_trainers = 3
    p = mp.Process(target=run_parameter_server, args=(0, num_trainers + 1))
    p.start()
    processes.append(p)
    # run num_trainers workers
    for i in range(num_trainers):
        p = mp.Process(
            target=run_worker,
            args=(
                i + 1,
                num_trainers + 1,
                train_loader,
                test_loader))
        p.start()
        processes.append(p)

    # Run to completeion.
    for p in processes:
        p.join()

    print("Script took {} seconds".format(time.time() - start))
