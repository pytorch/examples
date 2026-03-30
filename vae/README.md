# Basic VAE Example

This is an improved implementation of the paper [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

### Usage
Install the required dependencies:
```bash
pip install -r requirements.txt
```

To run the example, execute:
```bash
python main.py
```

If a hardware accelerator device is detected, the example will execute on the accelerator; otherwise, it will run on the CPU.

To force execution on the CPU, use `--no-accel` command line argument:

```bash
python main.py --no-accel
```

To run on a TPU via XLA, install `torch_xla` and use the `--xla` flag:

```bash
pip install torch_xla[tpu]
python main.py --xla
```

The main.py script accepts the following optional arguments:

```bash
--batch-size            input batch size for training (default: 128)
--epochs                number of epochs to train (default: 10)
--no-accel              disables accelerator
--xla                   enables XLA device (e.g. TPU). Requires torch_xla.
--seed                  random seed (default: 1)
--log-interval          how many batches to wait before logging training status
```

