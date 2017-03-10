# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Quickstart

Use of OpenNMT consists of four steps:

### 0) Download the data.

```wget https://s3.amazonaws.com/pytorch/examples/opennmt/data/onmt-data.tar && tar -xf onmt-data.tar```

### 1) Preprocess the data.

```python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo```

### 2) Train the model.

```python train.py -data data/demo-train.pt -save_model demo_model -gpus 0```

### 3) Translate sentences.

```python translate.py -gpu 0 -model demo_model_e13_*.pt -src data/src-test.txt -tgt data/tgt-test.txt -replace_unk -verbose -output demo_pred.txt```

### 4) Evaluate.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
perl multi-bleu.perl data/tgt-test.txt < demo_pred.txt
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

Data might not come as clean as the demo data. Here is a second example that uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the Multi30k data from the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
```

### 1) Preprocess the data.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi; perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
python preprocess.py -train_src data/multi30k/train.en.tok -train_tgt data/multi30k/train.de.tok -valid_src data/multi30k/val.en.tok -valid_tgt data/multi30k/val.de.tok -save_data data/multi30k
```

### 2) Train the model.

```python train.py -data data/multi30k-train.pt -save_model multi30k_model -gpus 0```

### 3) Translate sentences.

```python translate.py -gpu 0 -model multi30k_model_e13_*.pt -src data/multi30k/test.en.tok -tgt data/multi30k/test.de.tok -replace_unk -verbose -output multi30k_pred.txt```

### 4) Evaluate.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
perl multi-bleu.perl data/multi30k/test.de.tok < multi30k_pred.txt
```

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

- [onmt_model_en_de_200k](https://s3.amazonaws.com/pytorch/examples/opennmt/models/onmt_model_en_de_200k-4783d9c3.pt): An English-German translation model based on the 200k sentence dataset at [OpenNMT/IntegrationTesting](https://github.com/OpenNMT/IntegrationTesting/tree/master/data). Perplexity: 21.
- [onmt_model_en_fr_b1M](https://s3.amazonaws.com/pytorch/examples/opennmt/models/onmt_model_en_fr_b1M-261c69a7.pt): An English-French model trained on benchmark-1M. Perplexity: 4.85.

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
