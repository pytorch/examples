# /bin/bash
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to run. Default = 'example.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 4

# samples to run include:
# example.py

echo "Launching ${1:-example.py} with ${2:-4} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-4} ${1:-example.py}

