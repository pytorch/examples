# Sequence Labeling with BiRNN and CRF

This example trains a bidirectional RNN (LSTM or GRU) with CRF for part-of-speech tagging task. It uses the UDPOS dataset
 in `torchtext` and can be extended to other sequence labeling tasks by replacing the dataset accordingly.

The `train.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --emb_path EMB_PATH   path to pretrained embeddings
  --optimizer {sgd,adam,nesterov}
                        optimizer to use
  --sgd_momentum SGD_MOMENTUM
                        momentum for stochastic gradient descent
  --rnn {lstm,gru}      RNN type
  --rnn_dim RNN_DIM     RNN hidden state dimension
  --word_emb_dim WORD_EMB_DIM
                        word embedding dimension
  --char_emb_dim CHAR_EMB_DIM
                        char embedding dimension
  --char_rnn_dim CHAR_RNN_DIM
                        char RNN hidden state dimension
  --word_min_freq WORD_MIN_FREQ
                        minimum frequency threshold in word vocabulary
  --train_batch_size TRAIN_BATCH_SIZE
                        batch size for training phase
  --val_batch_size VAL_BATCH_SIZE
                        batch size for evaluation phase
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        early stopping patience
  --dropout_before_rnn DROPOUT_BEFORE_RNN
                        dropout rate on RNN inputs
  --dropout_after_rnn DROPOUT_AFTER_RNN
                        dropout rate on RNN outputs
  --lr LR               starting learning rate
  --lr_decay LR_DECAY   learning rate decay factor
  --min_lr MIN_LR       minimum learning rate
  --lr_shrink LR_SHRINK
                        learning rate reducing factor
  --lr_shrink_patience LR_SHRINK_PATIENCE
                        learning rate reducing patience
  --crf {none,small,large}
                        CRF type or no CRF
  --max_epochs MAX_EPOCHS
                        maximum training epochs
```
