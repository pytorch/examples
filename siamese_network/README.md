# Siamese Network Example
Siamese network for image similarity estimation.
The network is composed of two identical networks, one for each input.
The output of each network is concatenated and passed to a linear layer.
The output of the linear layer passed through a sigmoid function.
[FaceNet](https://arxiv.org/pdf/1503.03832.pdf) is a variant of the Siamese network.
This implementation varies from FaceNet as we use the `ResNet-18` model from
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) as our feature extractor.
In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.

### Usage

Install the required dependencies:
```bash
pip install -r requirements.txt
```

To run the example, execute:
```bahs
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

If a hardware accelerator device is detected, the example will execute on the accelerator; otherwise, it will run on the CPU.

To force execution on the CPU, use `--no-accel` command line argument:

```bash
python main.py --no-accel
```

Optionally, you can add the following arguments to customize your execution.

```bash
--batch-size            input batch size for training (default: 64)
--test-batch-size       input batch size for testing (default: 1000)
--epochs                number of epochs to train (default: 14)
--lr                    learning rate (default: 1.0)
--gamma                 learning rate step gamma (default: 0.7)
--no-accel              disables accelerator
--dry-run               quickly check a single pass
--seed                  random seed (default: 1)
--log-interval          how many batches to wait before logging training status
--save-model            Saving the current Model
```
