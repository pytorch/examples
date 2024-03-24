import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k, multi30k

# Turns an iterable into a generator
def _yield_tokens(iterable_data, tokenizer, src):

    # Iterable data stores the samples as (src, tgt) so this will help us select just one language or the other
    index = 0 if src else 1

    for data in iterable_data:
        yield tokenizer(data[index])

# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
def get_data(opts):

    src_lang = opts.src
    tgt_lang = opts.tgt

    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

    # Define a token "unkown", "padding", "beginning of sentence", and "end of sentence"
    special_symbols = {
        "<unk>":0,
        "<pad>":1,
        "<bos>":2,
        "<eos>":3
    }

    # Get training examples from torchtext (the multi30k dataset)
    train_iterator = Multi30k(split="train", language_pair=(src_lang, tgt_lang))
    valid_iterator = Multi30k(split="valid", language_pair=(src_lang, tgt_lang))

    # Grab a tokenizer for these languages
    src_tokenizer = get_tokenizer("spacy", src_lang)
    tgt_tokenizer = get_tokenizer("spacy", tgt_lang)

    # Build a vocabulary object for these languages
    src_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, src_tokenizer, src_lang),
        min_freq=1,
        specials=list(special_symbols.keys()),
        special_first=True
    )

    tgt_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, tgt_tokenizer, tgt_lang),
        min_freq=1,
        specials=list(special_symbols.keys()),
        special_first=True
    )

    src_vocab.set_default_index(special_symbols["<unk>"])
    tgt_vocab.set_default_index(special_symbols["<unk>"])

    # Helper function to sequentially apply transformations
    def _seq_transform(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # Function to add BOS/EOS and create tensor for input sequence indices
    def _tensor_transform(token_ids):
        return torch.cat(
            (torch.tensor([special_symbols["<bos>"]]),
             torch.tensor(token_ids),
             torch.tensor([special_symbols["<eos>"]]))
        )

    src_lang_transform = _seq_transform(src_tokenizer, src_vocab, _tensor_transform)
    tgt_lang_transform = _seq_transform(tgt_tokenizer, tgt_vocab, _tensor_transform)

    # Now we want to convert the torchtext data pipeline to a dataloader. We
    # will need to collate batches
    def _collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])
        return src_batch, tgt_batch

    # Create the dataloader
    train_dataloader = DataLoader(train_iterator, batch_size=opts.batch, collate_fn=_collate_fn)
    valid_dataloader = DataLoader(valid_iterator, batch_size=opts.batch, collate_fn=_collate_fn)

    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols

def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):

    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# A small test to make sure our data loasd in correctly
if __name__=="__main__":

    class Opts:
        def __init__(self):
            self.src = "en",
            self.tgt = "de"
            self.batch = 128

    opts = Opts()
    
    train_dl, valid_dl, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols = get_data(opts)

    print(f"{opts.src} vocab size: {len(src_vocab)}")
    print(f"{opts.src} vocab size: {len(tgt_vocab)}")

