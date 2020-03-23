### RPC-based distributed training

This is a basic example of RPC-based training that uses several trainers remotely train a model hosted on a server. 

To run the example locally, run the following command worker for the server and each worker you wish to spawn, in separate terminal windows:
`python rpc_parameter_server.py [world_size] [rank] [num_gpus]`. For example, for a master node with world size of 2, the command would be `python rpc_parameter_server.py 2 0 0`. The trainer can then be launched with the command `python rpc_parameter_server.py 2 1 0` in a separate window, and this will begin training with one server and a single trainer.

Note that for demonstration purposes, this example supports only between 0-2 GPUs, although the pattern can be extended to make use of additional GPUs.  

You can pass in the command line arguments `--master_addr=<address>` and `master_port=PORT` to indicate the address:port that the master worker is listening on. All workers will contact the master for rendezvous during worker discovery. By default, `master_addr` will be `localhost` and `master_port` will be 29500.
