#!/usr/bin/env bash
# This script runs through the code in each of the cpp examples.
# The purpose is just as an integration test, not to actually train models in any meaningful way.

# Optionally specify a comma separated list of examples to run.
# can be run as:
# ./run_cpp_examples.sh "get_libtorch,run_all,clean"
# To get libtorch, run all examples, and remove temporary/changed data files.

BASE_DIR=`pwd`"/"`dirname $0`
echo "BASE_DIR: $BASE_DIR"
EXAMPLES=`echo $1 | sed -e 's/ //g'`
HOME_DIR=$HOME
ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function get_libtorch() {
  echo "Getting libtorch"
  cd $HOME_DIR
  if [ ! -d "libtorch" ]; then
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
    unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
  fi

  if [ $? -eq 0 ]; then
    echo "Successfully downloaded and extracted libtorch"
    LIBTORCH_PATH="$HOME_DIR/libtorch"   # Store the LibTorch path in a variable.
    echo "LibTorch path: $LIBTORCH_PATH" # Print the LibTorch path
  else
    error "Failed to download or extract LibTorch"
  fi
}

function start() {
  EXAMPLE=${FUNCNAME[1]}
  cd $BASE_DIR/cpp/$EXAMPLE
  echo "Running example: $EXAMPLE"
}

function check_run_success() {
  if [ $? -eq 0 ]; then
    echo "Successfully ran $1"
  else
    echo "Failed to run $1"
    error "Failed to run $1"
    exit 1
  fi
}

function autograd() {
  start
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
  make
  if [ $? -eq 0 ]; then
    echo "Successfully built $EXAMPLE"
    ./$EXAMPLE # Run the executable
    check_run_success $EXAMPLE
  else
    error "Failed to build $EXAMPLE"
    exit 1
  fi
}

function custom-dataset() {
  start
  # Download the dataset and unzip it
  if [ ! -d "$BASE_DIR/cpp/$EXAMPLE/dataset" ]; then
    wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
    unzip caltech-101.zip
    cd caltech-101
    tar -xzf 101_ObjectCategories.tar.gz
    mv 101_ObjectCategories $BASE_DIR/cpp/$EXAMPLE/dataset
  fi
  # build the executable and run it
  cd $BASE_DIR/cpp/$EXAMPLE
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
  make
  if [ $? -eq 0 ]; then
    echo "Successfully built $EXAMPLE"
    cd $BASE_DIR/cpp/$EXAMPLE
    ./build/$EXAMPLE # Run the executable
    check_run_success $EXAMPLE
  else
    error "Failed to build $EXAMPLE"
    exit 1
  fi
}
function dcgan() {
  start
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
  make
  if [ $? -eq 0 ]; then
    echo "Successfully built $EXAMPLE"
    ./$EXAMPLE --epochs 5 # Run the executable with kNumberOfEpochs = 5
    check_run_success $EXAMPLE
  else
    error "Failed to build $EXAMPLE"
    exit 1
  fi
}

function mnist() {
  start
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
  make
  if [ $? -eq 0 ]; then
    echo "Successfully built $EXAMPLE"
    ./$EXAMPLE # Run the executable
    check_run_success $EXAMPLE
  else
    error "Failed to build $EXAMPLE"
    exit 1
  fi
}

function regression() {
  start
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
  make
  if [ $? -eq 0 ]; then
    echo "Successfully built $EXAMPLE"
    ./$EXAMPLE # Run the executable
    check_run_success $EXAMPLE
  else
    error "Failed to build $EXAMPLE"
    exit 1
  fi
}

function clean() {
  cd $BASE_DIR
  echo "Running clean to remove cruft"
  # Remove the build directories
  find . -type d -name 'build' -exec rm -rf {} +
  # Remove the libtorch directory
  rm -rf $HOME_DIR/libtorch
  rm -f $HOME_DIR/libtorch-shared-with-deps-latest.zip
  echo "Clean completed"
}

function run_all() {
  autograd
  custom-dataset
  dcgan
  mnist
  regression
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
  echo "Some examples failed:"
  printf "$ERRORS"
  exit 1
fi
