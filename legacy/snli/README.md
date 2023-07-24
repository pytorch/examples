# PyTorch-based NLI Training with SNLI

## 📝 Overview

This repository contains Python scripts to train a Natural Language Inference (NLI) model, specifically the `SNLIClassifier`, using the Stanford Natural Language Inference (SNLI) corpus. The trained model predicts textual entailment, identifying if a statement is entailed, contradicted, or neither by another statement.

## ⚙️ Dependencies

Install the necessary Python libraries with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```
torch
torchtext
spacy
```

## 💻 Usage

Start the training process with:

```bash
python train.py --lower --word-vectors [PATH_TO_WORD_VECTORS] --vector-cache [PATH_TO_VECTOR_CACHE] --epochs [NUMBER_OF_EPOCHS] --batch-size [BATCH_SIZE] --save-path [PATH_TO_SAVE_MODEL] --gpu [GPU_NUMBER]
```

## 🏋️‍♀️ Training

The script trains the model on mini-batches of data across a specified number of epochs. It saves the best-performing model on the validation set as a `.pt` file in the specified directory.

## 📚 Scripts

- `model.py`: Defines the `SNLIClassifier` model and auxiliary classes.
- `util.py`: Contains utility functions for directory creation and command-line argument parsing.

## 📣 Note

Ensure the `model.py` and `util.py` scripts are available in your working directory.