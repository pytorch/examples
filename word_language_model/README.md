# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided. 
The trained model can then be used by the generate script to generate new text.

```bash
python main.py -cuda  # Train an LSTM on ptb with cuda (cuDNN). Should reach perplexity of 116
python generate.py    # Generate samples from the trained LSTM model. 
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  -data DATA            Location of the data corpus
  -model MODEL          Type of recurrent net. RNN_TANH, RNN_RELU, LSTM, or
                        GRU.
  -emsize EMSIZE        Size of word embeddings
  -nhid NHID            Number of hidden units per layer.
  -nlayers NLAYERS      Number of layers.
  -lr LR                Initial learning rate.
  -clip CLIP            Gradient clipping.
  -maxepoch MAXEPOCH    Upper epoch limit.
  -batchsize BATCHSIZE  Batch size.
  -bptt BPTT            Sequence length.
  -seed SEED            Random seed.
  -cuda                 Use CUDA.
  -reportint REPORTINT  Report interval.
  -save SAVE            Path to save the final model.
```
