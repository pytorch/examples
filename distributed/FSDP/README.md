Note: FSDP1 is deprecated. Please follow [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [code examples](https://github.com/pytorch/examples/tree/main/distributed/FSDP2).

## FSDP1 T5



To run the T5 example with FSDP1 for text summarization:

## Get the wikihow dataset
```bash

sh download_dataset.sh

```

## Install the requirements:
~~~
pip install -r requirements.txt
~~~
## Ensure you are running a recent version of PyTorch:
see https://pytorch.org to install at least 1.12 and ideally a current nightly build. 

Start the training with Torchrun (adjust nproc_per_node to your GPU count):

```
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

```
