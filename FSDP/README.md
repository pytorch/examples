## FSDP T5

To run the T5 example with FSDP for text summarization:

## Get the wikihow dataset
1 - Create a folder called 'data' and cd to it
2 -  Download the two CSV files in [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) dataset as linked below:
~~~
wget https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv
wget https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowSep.csv
~~~

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
