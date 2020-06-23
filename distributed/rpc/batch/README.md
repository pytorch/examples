Batch RPC Server Example

This example shows how to create a batch RPC server using `torch.distributed.rpc`
package, where multiple clients uses RPC to run functions on the server and the
server processes multiple RPC requests together in a batch.

To try an example with three workers, try running the following three commands
to create three processes.


```
python server.py --name="s" --rank=0 --world_size=3
python client.py --name="c1" --rank=1 --world_size=3 --server_name="s"
python client.py --name="c2" --rank=2 --world_size=3 --server_name="s"
```
