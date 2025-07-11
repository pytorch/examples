
# Distributed Data Parallel (DDP) Applications with PyTorch

This guide demonstrates how to structure a distributed model training application for convenient multi-node launches using `torchrun`.

---

## Prerequisites

You should be familiar with:

- [PyTorch basics](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Writing distributed applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Distributed model training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

This tutorial uses the [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) class for data parallel training: multiple workers train the same global model on different data shards, compute local gradients, and synchronize them using AllReduce. In High-Performance Computing (HCP), this is called _Single Program Multiple Data_ (SPMD).

---

## Application Process Topologies

A Distributed Data Parallel (DDP) application can be executed on
multiple nodes where each node can consist of multiple GPU
devices. Each node in turn can run multiple copies of the DDP
application, each of which processes its models on multiple GPUs.

Let:
- _N_ = number of nodes
- _G_ = number of GPUs per node
- _W_ = **World Size** = total number of processes
- _L_ = **Local World Size** = processes per node

Each process has:
- **Local rank**: in `[0, L-1]`
- **Global rank**: in `[0, W-1]`

**Example:**
If you launch a DDP app on 2 nodes, each with 4 GPUs, and want each process to span 2 GPUs, the mapping is as follows:

![ProcessMapping](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

While there are quite a few ways to map processes to nodes, a good rule of thumb is to have one process span a single GPU. This enables the DDP application to have as many parallel reader streams as there are GPUs and in practice provides a good balance between I/O and computational costs. In the rest of this tutorial, we assume that the application follows this heuristic.

# Preparing and launching a DDP application

Independent of how a DDP application is launched, each process needs a mechanism to know its global and local ranks. Once this is known, all processes create a `ProcessGroup` that enables them to participate in collective communication operations such as AllReduce.

A convenient way to start multiple DDP processes and initialize all values needed to create a `ProcessGroup` is to use the [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) script provided with PyTorch.

---

## Sample Application

This example is based on the ["Hello, World" DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

The application calls the `spmd_main` entrypoint:

```python
if __name__ == "__main__":
    spmd_main()
```

In `spmd_main`, the process group is initialized using the Accelerator API. The rest of the rendezvous information comes from environment variables set by `torchrun`:

```python
def spmd_main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    acc = torch.accelerator.current_accelerator()
    vendor_backend = torch.distributed.get_default_backend_for_device(acc)
    torch.accelerator.set_device_index(rank)
    torch.distributed.init_process_group(backend=vendor_backend)

    demo_basic(rank)

    # Tear down the process group
    torch.distributed.destroy_process_group()
```

**Key points:**
- Each process reads its rank and world size from environment variables.
- The process group is initialized for distributed communication.

The training function, `demo_basic`, initializes the DDP model on the appropriate GPU:

```python
def demo_basic(rank):
    print(
        f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
        + f"world_size = {torch.distributed.get_world_size()}"
    )

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
```

---

## Launching the Application

```sh
torchrun --nnodes=1 --nproc_per_node=8 example.py
```

---

## Example Output

Expected output:

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
[238633] rank = 6, world_size = 8
[238628] rank = 1, world_size = 8
[238632] rank = 5, world_size = 8
[238634] rank = 7, world_size = 8
[238629] rank = 2, world_size = 8
[238630] rank = 3, world_size = 8
[238631] rank = 4, world_size = 8
[238627] rank = 0, world_size = 8
```

# Conclusions

As the author of a distributed data parallel application, your code needs to be aware of two types of resources: compute nodes and the GPUs within each node. The process of setting up bookkeeping to track how the set of GPUs is mapped to the processes of your application can be tedious and error-prone. We hope that by structuring your application as shown in this example and using `torchrun`, the mechanics of setting up distributed training can be significantly simplified.
