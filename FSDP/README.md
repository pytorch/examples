## FSDP T5

To run T5 example with FSDP for text_summarization:

## Get the wikihow dataset
1 - Create a folder called 'data' and cd to it
2 -  Download the two CSV files in [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) dataset as linked below:
~~~
wget https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358
~~~
~~~
wget https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag
~~~
direct links:
[wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) and 
[wikihowSep.csv](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag)

## install the requirements:
~~~
pip install -r requirements.txt
~~~
## ensure you are running a recent version of PyTorch:
see https://pytorch.org to install at least 1.12 and ideally a current nightly build. 

* Run the script
```
python FSDP_T5.py

```
For running T5 with Torchrun

```
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

```