import os

import torch.distributed.rpc as rpc
from batch import BatchServer, parse_args



def foo(x, y):
    return x + y


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    args = parse_args()
    rpc.init_rpc(args.name, rank=args.rank, world_size=args.world_size)
    server = BatchServer()
    server.bind(foo)
    rpc.shutdown()
