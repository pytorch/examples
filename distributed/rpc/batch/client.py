import os

import torch.distributed.rpc as rpc
from batch import BatchClient, parse_args


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    args = parse_args()
    rpc.init_rpc(args.name, rank=args.rank, world_size=args.world_size)
    client = BatchClient(args.server_name)
    y = args.rank * 100
    for x in range(5):
        print("Client {} got result {}".format(args.name, client.foo(x, y)))
    rpc.shutdown()
