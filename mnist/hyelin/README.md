# Basic MNIST Example

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help		show this help message and exit
  --batch-size N	input batch size for training (default: 64)
  --test-batch-size N	input batch size for testing (default: 1000)
  --epochs N		number of epochs to train (default: 14)
  --lr LR		learning rate (default: 1.0)
  --gamma M		learning rate step gamma (default: 0.7)
  --no-cuda		disables CUDA training
  --seed S		random seed (default: 1)
  --log-interval N	how many batches to wait before logging training status
  --save-model		for saving the current model
``` 


```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
