import os
import argparse
import torch
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs, Transformer


def main(args):
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(rank)
    vocab_size = 1024
    model_args = ModelArgs(
        n_layers=3,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=64,
        dropout_p=0,
    )
    model = Transformer(model_args)
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(10):
        x = torch.randint(0, vocab_size, (32, 32), device=device)
        loss = model(x).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FSDP2 example')
    parser.add_argument('--meta-init', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main(args)
