# Example of MNIST using RNN

## Motivation
Create pytorch example similar to Official Tensorflow Keras RNN example using MNIST [here](https://www.tensorflow.org/guide/keras/rnn) 

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
The main.py script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --batch_size          input batch_size for training (default:64)
  --testing_batch_size  input batch size for testing (default: 1000)
  --epochs EPOCHS       number of epochs to train (default: 14)
  --lr LR               learning rate (default: 0.1)
  --gamma               learning rate step gamma (default: 0.7)
  --cuda                enables CUDA training
  --xpu                 enables XPU training
  --mps                 enables macos GPU training
  --seed SEED           random seed (default: 1)
  --save_model          For saving the current Model
  --log_interval        how many batches to wait before logging training status
  --dry-run             quickly check a single pass
```