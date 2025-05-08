## FSDP2

To run FSDP2 on transformer model:

## Install the requirements:
~~~
pip install -r requirements.txt
~~~

## Ensure you are running a recent version of PyTorch:
see https://pytorch.org/get-started/locally/ to install at least 2.5 and ideally a current nightly build.

Start the training with `torchrun` Torchrun (adjust nproc_per_node to your GPU count):

```
torchrun --nproc_per_node 2 train.py
```
