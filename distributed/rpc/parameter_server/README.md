### RPC-based distributed training

This is a basic example of RPC-based training that can spawn several workers to train a model living on a server.

To run the example locally, run the following command format in a separate terminal window:
`python rpc_parameter_server [world_size] [rank]`. For example, for a master node with world_size of 2, the command would be `python rpc_parameter_server.py 2 0`.

You can pass in the command line arguments `--master_addr=<address>` and `master_port=PORT` to indicate the address:port that the master worker is listening on. All workers will contact the master for rendezvous during worker discovery.
