# /bin/bash
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to run. Default = 'main.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 2

# samples to run include:
# main.py

echo "Launching ${1:-main.py} with ${2:-2} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-2} --rdzv_id=101 ${1:-main.py} --dataset fake --dry-run
