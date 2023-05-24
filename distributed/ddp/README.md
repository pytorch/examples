# Launching and configuring distributed data parallel applications

In this tutorial we will demonstrate how to structure a distributed
model training application so it can be launched conveniently on
multiple nodes, each with multiple GPUs using PyTorch's distributed
[launcher script](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).

# Prerequisites

We assume you are familiar with [PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), the primitives it provides for [writing distributed applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html) as well as training [distributed models](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

The example program in this tutorial uses the
[`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) class for training models
in a _data parallel_ fashion: multiple workers train the same global
model by processing different portions of a large dataset, computing
local gradients (aka _sub_-gradients) independently and then
collectively synchronizing gradients using the AllReduce primitive. In
HPC terminology, this model of execution is called _Single Program
Multiple Data_ or SPMD since the same application runs on all
application but each one operates on different portions of the
training dataset.

# Application process topologies

A Distributed Data Parallel (DDP) application can be executed on
multiple nodes where each node can consist of multiple GPU
devices. Each node in turn can run multiple copies of the DDP
application, each of which processes its models on multiple GPUs.

Let _N_ be the number of nodes on which the application is running and
_G_ be the number of GPUs per node. The total number of application
processes running across all the nodes at one time is called the
**World Size**, _W_ and the number of processes running on each node
is referred to as the **Local World Size**, _L_.

Each application process is assigned two IDs: a _local_ rank in \[0,
_L_-1\] and a _global_ rank in \[0, _W_-1\].

To illustrate the terminology defined above, consider the case where a
DDP application is launched on two nodes, each of which has four
GPUs. We would then like each process to span two GPUs each. The
mapping of processes to nodes is shown in the figure below:

![ProcessMapping](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

While there are quite a few ways to map processes to nodes, a good
rule of thumb is to have one process span a single GPU. This enables
the DDP application to have as many parallel reader streams as there
are GPUs and in practice provides a good balance between I/O and
computational costs. In the rest of this tutorial, we assume that the
application follows this heuristic.

# Preparing and launching a DDP application

Independent of how a DDP application is launched, each process needs a
mechanism to know its global and local ranks. Once this is known, all
processes create a `ProcessGroup` that enables them to participate in
collective communication operations such as AllReduce.

A convenient way to start multiple DDP processes and initialize all
values needed to create a `ProcessGroup` is to use the distributed
`launch.py` script provided with PyTorch. The launcher can be found
under the `distributed` subdirectory under the local `torch`
installation directory. Here is a quick way to get the path of
`launch.py` on any operating system:

```sh
python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))"
```

This will print something like this:

```sh
/home/username/miniconda3/envs/pytorch/lib/python3.8/site-packages/torch/distributed/launch.py
```

When the DDP application is started via `launch.py`, it passes the world size, global rank, master address and master port via environment variables and the local rank as a command-line parameter to each instance.
To use the launcher, an application needs to adhere to the following convention:

1. It must provide an entry-point function for a _single worker_. For example, it should not launch subprocesses using `torch.multiprocessing.spawn`
2. It must use environment variables for initializing the process group.

For simplicity, the application can assume each process maps to a single GPU but in the next section we also show how a more general process-to-GPU mapping can be performed.

# Sample application

The sample DDP application in this repo is based on the "Hello, World" [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## Argument passing convention

The DDP application takes two command-line arguments:

1. `--local_rank`: This is passed in via `launch.py`
2. `--local_world_size`: This is passed in explicitly and is typically either $1$ or the number of GPUs per node.

The application parses these and calls the `spmd_main` entrypoint:

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    spmd_main(args.local_world_size, args.local_rank)
```

In `spmd_main`, the process group is initialized with just the backend (NCCL or Gloo). The rest of the information needed for rendezvous comes from environment variables set by `launch.py`:

```py
def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()
```

Given the local rank and world size, the training function, `demo_basic` initializes the `DistributedDataParallel` model across a set of GPUs local to the node via `device_ids`:

```py
def demo_basic(local_world_size, local_rank):

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    model = ToyModel().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()
```

The application can be launched via `launch.py` as follows on a 8 GPU node with one process per GPU:

```sh
python /path/to/launch.py --nnode=1 --node_rank=0 --nproc_per_node=8 example.py --local_world_size=8
```

and produces an output similar to the one shown below:

```sh
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[238627] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '0', 'WORLD_SIZE': '8'}
[238630] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '3', 'WORLD_SIZE': '8'}
[238628] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '1', 'WORLD_SIZE': '8'}
[238634] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '7', 'WORLD_SIZE': '8'}
[238631] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '4', 'WORLD_SIZE': '8'}
[238632] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '5', 'WORLD_SIZE': '8'}
[238629] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '2', 'WORLD_SIZE': '8'}
[238633] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '6', 'WORLD_SIZE': '8'}
[238633] world_size = 8, rank = 6, backend=nccl
[238628] world_size = 8, rank = 1, backend=nccl
[238629] world_size = 8, rank = 2, backend=nccl
[238631] world_size = 8, rank = 4, backend=nccl
[238630] world_size = 8, rank = 3, backend=nccl
[238632] world_size = 8, rank = 5, backend=nccl
[238634] world_size = 8, rank = 7, backend=nccl
[238627] world_size = 8, rank = 0, backend=nccl
[238633] rank = 6, world_size = 8, n = 1, device_ids = [6]
[238628] rank = 1, world_size = 8, n = 1, device_ids = [1]
[238632] rank = 5, world_size = 8, n = 1, device_ids = [5]
[238634] rank = 7, world_size = 8, n = 1, device_ids = [7]
[238629] rank = 2, world_size = 8, n = 1, device_ids = [2]
[238630] rank = 3, world_size = 8, n = 1, device_ids = [3]
[238631] rank = 4, world_size = 8, n = 1, device_ids = [4]
[238627] rank = 0, world_size = 8, n = 1, device_ids = [0]
```

Similarly, it can be launched with a single process that spans all 8 GPUs using:

```sh
python /path/to/launch.py --nnode=1 --node_rank=0 --nproc_per_node=1 example.py --local_world_size=1
```

that in turn produces the following output

```sh
[262816] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '0', 'WORLD_SIZE': '1'}
[262816]: world_size = 1, rank = 0, backend=nccl
[262816] rank = 0, world_size = 1, n = 8, device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
```

# Conclusions

As the author of a distributed data parallel application, your code needs to be aware of two types of resources: compute nodes and the GPUs within each node. The process of setting up bookkeeping to track how the set of GPUs is mapped to the processes of your application can be tedious and error-prone. We hope that by structuring your application as shown in this example and using the launcher, the mechanics of setting up distributed training can be significantly simplified.
