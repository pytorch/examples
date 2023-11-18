torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 --rdzv_endpoint="localhost:5973" sequence_parallel_example.py
