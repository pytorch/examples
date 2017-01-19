# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Quickstart

OpenNMT consists of three commands:

1) Preprocess the data.

```python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo```

2) Train the model.

```python train.py -data data/demo-train.pt -save_model model -cuda```

3) Translate sentences.

```python translate.py -cuda -model model_e13_*.pt -src data/src-test.txt -tgt data/tgt-test.txt -replace_unk -verbose```

## Release Notes

The following OpenNMT features are implemented:

- multi-layer bidirectional RNNs with attention and dropout
- data preprocessing
- saving and loading from checkpoints
- inference (translation) with batching and beam search

Not yet implemented:

- word features
- multi-GPU
- residual connections

## Performance

With default parameters on a single Maxwell GPU, this version runs about 70% faster than the Lua torch OpenNMT. The improved performance comes from two main sources:

- CuDNN is used for the encoder (although not for the decoder, since it can't handle attention)
- The decoder softmax layer is batched to efficiently trade off CPU vs. memory efficiency; this can be tuned with the -max_generator_batches parameter.
