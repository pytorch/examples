from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k, multi30k

def _yield_tokens(iterable_data, tokenizer, src):

    # Iterable data stores the samples as (src, tgt) so this will help us select just one language or the other
    index = 0 if src else 1

    for data in iterable_data:
        yield tokenizer(data[index])


def get_data(src_lang, tgt_lang):

    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

    # Define a token "unkown", "padding", "beginning of sentence", and "end of sentence"
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    train_iterator = Multi30k(split="train", language_pair=(src_lang, tgt_lang))
    valid_iterator = Multi30k(split="valid", language_pair=(src_lang, tgt_lang))

    src_tokenizer = get_tokenizer("spacy", src_lang)
    tgt_tokenizer = get_tokenizer("spacy", tgt_lang)

    src_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, src_tokenizer, src_lang),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )

    tgt_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, tgt_tokenizer, tgt_lang),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )

    src_vocab.set_default_index(UNK_IDX)
    tgt_vocab.set_default_index(UNK_IDX)

    return train_iterator, valid_iterator, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab
    

if __name__=="__main__":

    src = "en"
    tgt = "de"
    
    train_iterator, valid_iterator, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = get_data(src, tgt)

    train_size = sum(1 for _ in train_iterator)
    valid_size = sum(1 for _ in valid_iterator)

    print(f"Training examples loaded: {train_size} examples")
    print(f"Validation examples loaded: {valid_size} examples")
    print(f"{src} vocab size: {len(src_vocab)}")
    print(f"{tgt} vocab size: {len(tgt_vocab)}")

