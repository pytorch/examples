#!/usr/bin/env bash
# This script contains utility functions and initialize exmaple scripts.
# Eg: run_python_examples.sh, run_distributed_examples.sh

BASE_DIR="$(pwd)/$(dirname $0)"
EXAMPLES=$(echo $1 | sed -e 's/ //g')

# Redirect 'python' calls to 'python3'
python() {
    command python3 "$@"
}

ERRORS=${ERRORS-""}

function error() {
  ERR=$1
  if [ "" == "$ERRORS" ]; then
    ERRORS="$ERR"
  else
    ERRORS="$ERRORS\n$ERR"
  fi
}

function install_deps() {
  EXAMPLE_NAME=$1
  echo "] $EXAMPLE_NAME: installing requirements"
  [[ -f requirements.txt ]] || { error "requirements.txt not found; skipping"; return; }
  cat requirements.txt | \
    sort -u | \
    # testing the installed version of torch, so don't pip install it.
    grep -vE '^torch$' | \
    pip install -r /dev/stdin || \
    { error "failed to install dependencies"; exit 1; }
}

function start() {
  EXAMPLE_NAME=${FUNCNAME[1]}
  cd $BASE_DIR/$EXAMPLE_NAME
  install_deps $EXAMPLE_NAME
  echo "] $EXAMPLE_NAME: running"
}
