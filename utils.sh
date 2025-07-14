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

function start() {
  EXAMPLE=$1
  cd $BASE_DIR/$EXAMPLE || { error "$EXAMPLE: no such example"; return 1; }
  echo "Install dependencies for $EXAMPLE"
  # Setting VIRTUAL_ENV=.venv externally will create uv virtual environment
  # for each sample in start() and remove it in stop(). Note that this environment
  # variable also forces other uv commands such as `uv pip...` and `uv run...` to
  # use the specified environment.
  if [ "$VIRTUAL_ENV" = ".venv" ]; then
    uv venv || { error "$EXAMPLE: failed to create virtual environment"; return 1; }
  fi
  uv pip install -r requirements.txt $PIP_INSTALL_ARGS || { error "$EXAMPLE: failed to install requirements"; return 1; }
  echo "Running example: $EXAMPLE"
}

function stop() {
  EXAMPLE=$1
  if [ "$VIRTUAL_ENV" = ".venv" ]; then
    cd $BASE_DIR/$EXAMPLE && rm -rf .venv
  fi
}

function run() {
  EXAMPLE=$1
  if start $EXAMPLE; then
    # drop trailing slash (occurs due to auto completion in bash interactive mode)
    # replace slashes with underscores: this allows to call nested examples
    EXAMPLE_FN=$(echo $EXAMPLE | sed "s@/\$@@" | sed 's@/@_@g')
    $EXAMPLE_FN
  fi
  stop $EXAMPLE
}
