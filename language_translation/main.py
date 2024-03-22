from src.model import Translator
from src.data import get_data
from argparse import ArgumentParser


def inference(opts):
    pass

def main(opts):
    train_iterator, valid_iterator, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = get_data(src, tgt)


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training program",
    )

    # Inference mode
    parser.add_argument("--inference", type=bool, default=False,
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")

    # Translation settings
    parser.add_argument("--src", type=str, default="en",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="de",
                        help="Target language (translating TO this language)")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=1,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=1,
                        help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feedforward dimensionality")

    args = parser.parse_args()

    if args.inference:
        inference(args)
    else:
        main(args)
