#!/usr/bin/env bash
#
# This script runs through the code in each of the python examples.
# The purpose is just as an integration test, not to actually train models in any meaningful way.
# For that reason, most of these set epochs = 1 and --dry-run.
#
# Optionally specify a comma separated list of examples to run.
# can be run as:
# ./run_python_examples.sh "install_deps,run_all,clean"
# to pip install dependencies (other than pytorch), run all examples, and remove temporary/changed data files.
# Expects pytorch, torchvision to be installed.

source ./init_examples.sh

function distributed() {
    start
    bash tensor_parallelism/run_example.sh tensor_parallelism/tensor_parallel_example.py || error "tensor parallel example failed"
    bash tensor_parallelism/run_example.sh tensor_parallelism/sequence_parallel_example.py || error "sequence parallel example failed"
    bash tensor_parallelism/run_example.sh tensor_parallelism/fsdp_tp_example.py || error "2D parallel example failed"
    python ddp/main.py || error "ddp example failed"
}

function clean() {
  cd $BASE_DIR
  echo "running clean to remove cruft"
}

function run_all() {
  distributed
}

# by default, run all examples
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    $i
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