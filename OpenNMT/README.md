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
