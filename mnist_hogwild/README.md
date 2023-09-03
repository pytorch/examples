# MNIST Hogwild Example

```bash
pip install -r requirements.txt
python main.py
```

The main.py script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --batch_size          input batch_size for training (default:64)
  --testing_batch_size  input batch size for testing (default: 1000)
  --epochs EPOCHS       number of epochs to train (default: 1000)
  --lr LR               learning rate (default: 0.03)
  --momentum            SGD momentum (default: 0.5)
  --seed SEED           random seed (default: 1)
  --mps                 enables macos GPU training
  --save_model          For saving the current Model
  --log_interval        how many batches to wait before logging training status
  --num_process         how many training processes to use (default: 2)
  --cuda                enables CUDA training
  --dry-run             quickly check a single pass
  --save-model          For Saving the current Model
```
