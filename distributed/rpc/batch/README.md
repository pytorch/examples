# Examples For Asynchronous RPC User Functions

This folder contains two examples for [`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution):

1. Synchronized Batch Update Parameter Server: uses `@rpc.functions.async_execution`
   for parameter update and retrieving. This serves as a simple starter example
   for batch RPC.
   ```
   pip install -r requirements.txt
   python parameter_server.py
   ```
2. Multi-Observer with Batch-Processing Agent: uses `@rpc.functions.async_execution`
   to run multiple observed states through the policy to get actions.
   ```
   pip install -r requirements.txt
   python reinforce.py
   ```
