from time import time # Track how long an epoch takes
import os # Creating and finding files/directories
import logging # Logging tools
from datetime import date # Logging the date for model versioning

import torch # For ML
from tqdm import tqdm # For fancy progress bars

from src.model import Translator # Our model
from src.data import get_data, create_mask, generate_square_subsequent_mask # Loading data and data preprocessing
from argparse import ArgumentParser # For args

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):

    # Move to device
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    # Encode input
    memory = model.encode(src, src_mask)

    # Output will be stored here
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # For each element in our translation (which could range up to the maximum translation length)
    for _ in range(max_len-1):

        # Decode the encoded representation of the input
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)

        # Reshape
        out = out.transpose(0, 1)

        # Covert to probabilities and take the max of these probabilities
        prob = model.ff(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Now we have an output which is the vector representation of the translation
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys

# Opens an user interface where users can translate an arbitrary sentence
def inference(opts):

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    _, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    # Load in weights
    model.load_state_dict(torch.load(opts.model_path))

    # Set to inference
    model.eval()

    # Accept input and keep translating until they quit
    while True:
        print("> ", end="")

        sentence = input()

        # Convert to tokens
        src = src_transform(sentence).view(-1, 1)
        num_tokens = src.shape[0]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # Decode
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=num_tokens+5, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"]
        ).flatten()

        # Convert to list of tokens
        output_as_list = list(tgt_tokens.cpu().numpy())

        # Convert tokens to words
        output_list_words = tgt_vocab.lookup_tokens(output_as_list)

        # Remove special tokens and convert to string
        translation = " ".join(output_list_words).replace("<bos>", "").replace("<eos>", "")

        print(translation)

# Train the model for 1 epoch
def train(model, train_dl, loss_fn, optim, special_symbols, opts):

    # Object for accumulating losses
    losses = 0

    # Put model into inference mode
    model.train()
    for src, tgt in tqdm(train_dl, ascii=True):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        # Step weights
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()

        if opts.dry_run:
            break

    # Return the average loss
    return losses / len(list(train_dl))

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn, special_symbols):
    
    # Object for accumulating losses
    losses = 0

    # Turn off gradients a moment
    model.eval()

    for src, tgt in tqdm(valid_dl):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    # Return the average loss
    return losses / len(list(valid_dl))

# Train the model
def main(opts):

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)

    # This prints it to the screen as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")
    logging.info(f"Using device: {DEVICE}")

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)

    logging.info("Loaded data")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    logging.info("Model created... starting training!")

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss = train(model, train_dl, loss_fn, optim, special_symbols, opts)
        epoch_time = time() - start_time
        val_loss   = validate(model, valid_dl, loss_fn, special_symbols)

        # Once training is done, we want to save out the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")

        logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {val_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds")

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training and inference",
    )

    # Inference mode
    parser.add_argument("--inference", action="store_true",
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")

    # Translation settings
    parser.add_argument("--src", type=str, default="de",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="en",
                        help="Target language (translating TO this language)")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="cpu",
                        help="Batch size")
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=5,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=5,
                        help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./" + str(date.today()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")

    if args.inference:
        inference(args)
    else:
        main(args)
