## FSDP T5

To run T5 example with FSDP for text_summerization
* Download the two CSV files in [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) dataset as linked below:
* [wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) and [wikihowSep.csv](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag)
* Create "data" folder and place the two files in the folder.
* Run the script
```
python FSDP_T5.py

```
For running T5 with Torchrun

```
torchrun --nnodes 1 --nproc_per_node 4  main.py

```