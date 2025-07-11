#!/bin/bash
# Usage: bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to run. Default = 'main.py'
# num_gpus = num local gpus to use. Default = 16

# samples to run include:
# main.py

echo "Launching ${1:-main.py} with ${2:-16} gpus"
torchrun --standalone --nproc_per_node=${2:-16} ${1:-main.py}
