# minGPT-DDP

Code accompanying the tutorial at https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html for training a GPT-like model with Distributed Data Parallel (DDP) in PyTorch. 

Files marked with an asterisk (*) are adapted from the minGPT repo (https://github.com/karpathy/minGPT). 

- [trainer.py](mingpt/trainer.py) includes the Trainer class that runs the distributed training iterations on the model with the provided dataset.
- [model.py *](mingpt/model.py) defines the model architecture.
- [char_dataset.py *](mingpt/char_dataset.py) contains the `Dataset`class for a character-level dataset.
- [gpt2_train_cfg.yaml](mingpt/gpt2_train_cfg.yaml) contains the configurations for data, model, optimizer and training run.
- [main.py](mingpt/main.py) is the entry point to the trainig job. It sets up the DDP process group, reads all the configurations and runs the training job.
- [slurm/](mingpt/slurm) contains files for setting up an AWS cluster and the slurm script to run multinode training.