#!/usr/bin/env bash
#
# This script runs through the code in each of the python examples.
# The purpose is just as an integration test, not to actually train models in any meaningful way.
# For that reason, most of these set epochs = 1 and --dry-run.
#
# Optionally specify a comma separated list of examples to run. Can be run as:
# * To run all examples:
#   ./run_distributed_examples.sh
# * To run specific example:
#   ./run_distributed_examples.sh "distributed/tensor_parallelism,distributed/ddp"
#
# To test examples on CUDA accelerator, run as:
#   USE_CUDA=True ./run_distributed_examples.sh
#
# Script requires uv to be installed. When executed, script will install prerequisites from
# `requirements.txt` for each example. If ran within activated virtual environment (uv venv,
# python -m venv, conda) this might reinstall some of the packages. To change pip installation
# index or to pass additional pip install options, run as:
#   PIP_INSTALL_ARGS="--pre -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html" \
#     ./run_python_examples.sh
#
# To force script to create virtual environment for each example, run as:
#   VIRTUAL_ENV=".venv" ./run_distributed_examples.sh
# Script will remove environments it creates in a teardown step after execution of each example.

BASE_DIR="$(pwd)/$(dirname $0)"
source $BASE_DIR/utils.sh

USE_CUDA=${USE_CUDA:-False}
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

function distributed_tensor_parallelism() {
    uv run bash run_example.sh tensor_parallel_example.py || error "tensor parallel example failed"
    uv run bash run_example.sh sequence_parallel_example.py || error "sequence parallel example failed"
    uv run bash run_example.sh fsdp_tp_example.py || error "2D parallel example failed"
}

function distributed_FSDP2() {
    uv run bash run_example.sh example.py || error "FSDP2 example failed"
}

function distributed_ddp() {
    uv run bash run_example.sh example.py || error "ddp example failed"
}

function distributed_minGPT-ddp() {
  uv run bash run_example.sh mingpt/main.py || error "minGPT example failed"
}

function distributed_rpc_ddp_rpc() {
    uv run main.py || error "ddp_rpc example failed"
}

function distributed_rpc_rnn() {
    uv run main.py || error "rpc_rnn example failed"
}

function run_all() {
  run distributed/tensor_parallelism
  run distributed/ddp
  run distributed/minGPT-ddp
  run distributed/rpc/ddp_rpc
  run distributed/rpc/rnn
}

# by default, run all examples
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    run $i
    echo "Finished $i, status $?"
  done
fi

if [ "" == "$ERRORS" ]; then
  echo "Completed successfully with status $?"
else
  echo "Some distributed examples failed:"
  printf "$ERRORS\n"
  #Exit with error (0-255) in case of failure in one of the tests.
  exit 1

fi
