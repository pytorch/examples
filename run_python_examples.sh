#!/usr/bin/env bash
#
# This script runs through the code in each of the python examples.
# The purpose is just as an integrtion test, not to actually train
# models in any meaningful way. For that reason, most of these set
# epochs = 1 and --dry-run.
#
# Optionally specify a comma separated list of examples to run.
# can be run as:
# ./run_python_examples.sh "install_deps,run_all,clean"
# to pip install dependencies (other than pytorch), run all examples,
# and remove temporary/changed data files.
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
  if [ ! -d "lsun" ]; then
    echo "cloning repo to get lsun dataset"
    git clone https://github.com/fyu/lsun || { error "couldn't clone lsun repo needed for dcgan";  return; }
  fi
  # 'classroom' much smaller than the default 'bedroom' dataset.
  DATACLASS="classroom"
  if [ ! -d "lsun/${DATACLASS}_train_lmdb" ]; then
    pushd lsun
    python download.py -c $DATACLASS || { error "couldn't download $DATACLASS for dcgan";  return; }
    unzip ${DATACLASS}_train_lmdb.zip || { error "couldn't unzip $DATACLASS"; return; }
    popd
  fi
  python main.py --dataset lsun --dataroot lsun --classes $DATACLASS --niter 1 $CUDA_FLAG --dry-run || error "dcgan failed"
}

function fast_neural_style() {
  start
  if [ ! -d "saved_models" ]; then
    echo "downloading saved models for fast neural style"
    python download_saved_models.py
  fi
  test -d "saved_models" || { error "saved models not found"; return; }

  echo "running fast neural style model"
  python neural_style/neural_style.py eval --content-image images/content-images/amber.jpg --model saved_models/candy.pth --output-image images/output-images/amber-candy.jpg --cuda $CUDA || error "neural_style.py failed"
}

function imagenet() {
  start
  if [[ ! -d "sample/val" || ! -d "sample/train" ]]; then
    mkdir -p sample/val/n
    mkdir -p sample/train/n
    wget "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
    mv Socks-clinton.jpg sample/train/n
    cp sample/train/n/* sample/val/n/
  fi
  python main.py --epochs 1 sample/ || error "imagenet example failed"
}

function mnist() {
  start
  python main.py --epochs 1 --dry-run || error "mnist example failed"
}

function mnist_hogwild() {
  start
  python main.py --epochs 1 --dry-run $CUDA_FLAG || error "mnist hogwild failed"
}

function regression() {
  start
  python main.py --epochs 1 $CUDA_FLAG || error "regression failed"
}

function reinforcement_learning() {
  start
  python reinforce.py || error "reinforcement learning failed"
}

function snli() {
  start
  echo "installing 'en' model if not installed"
  python -m spacy download en || { error "couldn't download 'en' model needed for snli";  return; }
  echo "training..."
  python train.py --epochs 1 --dev_every 1 --no-bidirectional --dry-run || error "couldn't train snli"
}

function super_resolution() {
  start
  python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 1 --lr 0.001  || error "super resolution failed"
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

function word_language_model() {
  start
  python main.py --epochs 1 --dry-run $CUDA_FLAG || error "word_language_model failed"
}

function clean() {
  cd $BASE_DIR
  echo "running clean to remove cruft"
  rm -rf dcgan/_cache_lsun_classroom_train_lmdb \
    dcgan/fake_samples_epoch_000.png dcgan/lsun/ \
    dcgan/_cache_lsunclassroomtrainlmdb \
    dcgan/netD_epoch_0.pth dcgan/netG_epoch_0.pth \
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
    word_language_model/model.pt || error "couldn't clean up some files"

  git checkout fast_neural_style/images/output-images/amber-candy.jpg || error "couldn't clean up fast neural style image"
}

function run_all() {
  # cpp
  dcgan
  # distributed
  fast_neural_style
  imagenet
  mnist
  mnist_hogwild
  regression
  reinforcement_learning
  snli
  super_resolution
  time_sequence_prediction
  vae
  word_language_model
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
fi
