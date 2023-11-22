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

BASE_DIR=`pwd`"/"`dirname $0`
EXAMPLES=`echo $1 | sed -e 's/ //g'`

USE_CUDA=$(python -c "import torchvision, torch; print(torch.cuda.is_available())")
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

ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function install_deps() {
  echo "installing requirements"
  cat $BASE_DIR/*/requirements.txt | \
    sort -u | \
    # testing the installed version of torch, so don't pip install it.
    grep -vE '^torch$' | \
    pip install -r /dev/stdin || \
    { error "failed to install dependencies"; exit 1; }
}

function start() {
  EXAMPLE=${FUNCNAME[1]}
  cd $BASE_DIR/$EXAMPLE
  echo "Running example: $EXAMPLE"
}

function dcgan() {
  start
  python main.py --dataset fake $CUDA_FLAG --mps --dry-run || error "dcgan failed"
}

function distributed() {
    start
    python tensor_parallelism/tensor_parallel_example.py || error "tensor parallel example failed"
    python tensor_parallelism/sequence_parallel_example.py || error "sequence parallel example failed"
    python tensor_parallelism/fsdp_tp_example.py || error "2D parallel example failed"
    python ddp/main.py || error "ddp example failed"
}

function fast_neural_style() {
  start
  if [ ! -d "saved_models" ]; then
    echo "downloading saved models for fast neural style"
    python download_saved_models.py
  fi
  test -d "saved_models" || { error "saved models not found"; return; }

  echo "running fast neural style model"
  python neural_style/neural_style.py eval --content-image images/content-images/amber.jpg --model saved_models/candy.pth --output-image images/output-images/amber-candy.jpg --cuda $CUDA --mps || error "neural_style.py failed"
}

function imagenet() {
  start
  if [[ ! -d "sample/val" || ! -d "sample/train" ]]; then
    mkdir -p sample/val/n
    mkdir -p sample/train/n
    curl -O "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
    mv Socks-clinton.jpg sample/train/n
    cp sample/train/n/* sample/val/n/
  fi
  python main.py --epochs 1 sample/ || error "imagenet example failed"
}

function mnist() {
  start
  python main.py --epochs 1 --dry-run || error "mnist example failed"
}
function mnist_forward_forward() {
  start
  python main.py --epochs 1 --no_mps --no_cuda || error "mnist forward forward failed"

}
function mnist_hogwild() {
  start
  python main.py --epochs 1 --dry-run $CUDA_FLAG || error "mnist hogwild failed"
}

function mnist_rnn() {
  start
  python main.py --epochs 1 --dry-run || error "mnist rnn example failed"
}

function regression() {
  start
  python main.py --epochs 1 $CUDA_FLAG || error "regression failed"
}

function siamese_network() {
  start
  python main.py --epochs 1 --dry-run || error "siamese network example failed"
}

function reinforcement_learning() {
  start
  python reinforce.py || error "reinforcement learning reinforce failed"
  python actor_critic.py || error "reinforcement learning actor_critic failed"
}

function snli() {
  start
  echo "installing 'en' model if not installed"
  python -m spacy download en || { error "couldn't download 'en' model needed for snli";  return; }
  echo "training..."
  python train.py --epochs 1 --dev_every 1 --no-bidirectional --dry-run || error "couldn't train snli"
}

function fx() {
  start
  # python custom_tracer.py || error "fx custom tracer has failed" UnboundLocalError: local variable 'tabulate' referenced before assignment
  python invert.py || error "fx invert has failed"
  python module_tracer.py || error "fx module tracer has failed"
  python primitive_library.py || error "fx primitive library has failed"
  python profiling_tracer.py || error "fx profiling tracer has failed"
  python replace_op.py || error "fx replace op has failed"
  python subgraph_rewriter_basic_use.py || error "fx subgraph has failed"
  python wrap_output_dynamically.py || error "vmap output dynamically has failed"
}

function super_resolution() {
  start
  python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 1 --lr 0.001 --mps || error "super resolution failed"
}

function time_sequence_prediction() {
  start
  python generate_sine_wave.py || { error "generate sine wave failed";  return; }
  python train.py --steps 2 || error "time sequence prediction training failed"
}

function vae() {
  start
  python main.py --epochs 1 || error "vae failed"
}

function vision_transformer() {
  start
  python main.py --epochs 1 --dry-run || error "vision transformer example failed"
}

function word_language_model() {
  start
  python main.py --epochs 1 --dry-run $CUDA_FLAG --mps || error "word_language_model failed"
}

function gcn() {
  start
  python main.py --epochs 1 --dry-run || error "graph convolutional network failed"
}

function gat() {
  start
  python main.py --epochs 1 --dry-run || error "graph attention network failed"
}

function clean() {
  cd $BASE_DIR
  echo "running clean to remove cruft"
  rm -rf dcgan/fake_samples_epoch_000.png \
    dcgan/netD_epoch_0.pth \
    dcgan/netG_epoch_0.pth \
    dcgan/real_samples.png \
    fast_neural_style/saved_models.zip \
    fast_neural_style/saved_models/ \
    imagenet/checkpoint.pth.tar \
    imagenet/lsun/ \
    imagenet/model_best.pth.tar \
    imagenet/sample/ \
    snli/.data/ \
    snli/.vector_cache/ \
    snli/results/ \
    super_resolution/dataset/ \
    super_resolution/model_epoch_1.pth \
    time_sequence_prediction/predict*.pdf \
    time_sequence_prediction/traindata.pt \
    word_language_model/model.pt \
    gcn/cora/ \
    gat/cora/ || error "couldn't clean up some files"

  git checkout fast_neural_style/images/output-images/amber-candy.jpg || error "couldn't clean up fast neural style image"
}

function run_all() {
  # cpp
  dcgan
  distributed
  fast_neural_style
  imagenet
  mnist
  mnist_forward_forward
  mnist_hogwild
  mnist_rnn
  regression
  reinforcement_learning
  siamese_network
  super_resolution
  time_sequence_prediction
  vae
  vision_transformer
  word_language_model
  fx
  gcn
  gat
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
  #Exit with error (0-255) in case of failure in one of the tests.
  exit 1

fi
