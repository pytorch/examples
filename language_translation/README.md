# Language Translation

This example shows how one might use transformers for language translation. In particular, this implementation is loosely based on the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762)

## Requirements

We will need a tokenizer for our languages. Torchtext does include a tokenizer for English, but unfortunately, we will need more languages then that. We can get these tokenizers via ```spacy```

```
python -m spacy download <language>
python -m spacy download en
python -m spacy download de
```

Spacy supports language support can be found [here](https://spacy.io/usage/models). This example will default to English and German.

Torchtext is also required:
```
pip install torchtext
```

## Usage


