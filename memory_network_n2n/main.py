import argparse
import os
import sys

import torch
from net import N2N
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from util import long_tensor_type
from util import process_data, get_batch_from_batch_list, generate_batches


def train_network(train_batch_list, val_batch_list, test_batch_list, train, val, test, vocab_size, story_size,
                  save_model_path, args):
    net = N2N(args.batch_size, args.embed_size, vocab_size, args.hops, story_size=story_size)
    if torch.cuda.is_available() and args.cuda == 1:
        net = net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.debug is True:
        print("TRAINABLE PARAMETERS IN THE NETWORK: ", list(net.parameters()))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    running_loss = 0.0

    train_batches, val_batches, test_batches = generate_batches(train_batch_list, val_batch_list, test_batch_list,
                                                                train, val, test)

    best_val_acc_yet = 0.0
    for current_epoch in range(args.epochs):
        current_len = 0
        current_correct = 0
        for batch in train_batches:
            idx_out, idx_true, out = epoch(batch, net)
            loss = criterion(out, idx_true)
            loss.backward()

            clip_grad_norm(net.parameters(), 40)
            running_loss += loss
            current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
            optimizer.step()
            optimizer.zero_grad()

        if current_epoch % args.log_epochs == 0:
            accuracy = 100 * (current_correct / current_len)
            val_acc = calculate_loss_and_accuracy(net, val_batches)
            print("Epochs: {}, Train Accuracy: {}, Loss: {}, Val_Acc:{}".format(current_epoch, accuracy,
                                                                                running_loss.data[0],
                                                                                val_acc))
            if best_val_acc_yet <= val_acc:
                torch.save(net.state_dict(), save_model_path)
                best_val_acc_yet = val_acc

        if current_epoch % args.anneal_epoch == 0 and current_epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / args.anneal_factor
        running_loss = 0.0


def epoch(batch, net):
    story_batch = batch[0]
    query_batch = batch[1]
    answer_batch = batch[2]
    A = Variable(torch.stack(answer_batch, dim=0), requires_grad=False).type(long_tensor_type)
    _, idx_true = torch.max(A, 1)
    idx_true = torch.squeeze(idx_true)

    S = torch.stack(story_batch, dim=0)
    Q = torch.stack(query_batch, dim=0)
    out = net(S, Q)

    _, idx_out = torch.max(out, 1)
    return idx_out, idx_true, out


def update_counts(current_correct, current_len, idx_out, idx_true):
    batch_len, correct = count_predictions(idx_true, idx_out)
    current_len += batch_len
    current_correct += correct
    return current_correct, current_len


def count_predictions(labels, predicted):
    batch_len = len(labels)
    correct = (predicted == labels).sum().data[0]
    return batch_len, correct


def calculate_loss_and_accuracy(net, batches):
    current_len = 0
    current_correct = 0
    for batch in batches:
        idx_out, idx_true, out = epoch(batch, net)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)
    return 100 * (current_correct / current_len)


def eval_network(vocab_size, story_size, model, test_batches, test, EMBED_SIZE=50, batch_size=2,
                 depth=1, cuda=0):
    print("Evaluating")
    net = N2N(batch_size, EMBED_SIZE, vocab_size, depth, story_size=story_size)
    net.load_state_dict(torch.load(model))
    if torch.cuda.is_available() and cuda == 1:
        net = net.cuda()
    test_batches = get_batch_from_batch_list(test_batches, test)

    current_len = 0
    current_correct = 0

    for batch in test_batches:
        idx_out, idx_true, out = epoch(batch, net)
        current_correct, current_len = update_counts(current_correct, current_len, idx_out, idx_true)

    accuracy = 100 * (current_correct / current_len)
    print("Accuracy : ", str(accuracy))


def check_paths(args):
    try:
        if not os.path.exists(args.saved_model_dir):
            os.makedirs(args.saved_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def model_path(args):
    if args.joint_training == 1:
        saved_model_filename = "joint_model.model"
    else:
        saved_model_filename = str(args.task_number) + "_model.model"
    saved_model_path = os.path.join(args.saved_model_dir, saved_model_filename)
    return saved_model_path


def main():
    arg_parser = argparse.ArgumentParser(description="parser for End-to-End Memory Networks")

    arg_parser.add_argument("--train", type=int, default=1)
    arg_parser.add_argument("--epochs", type=int, default=100,
                            help="number of training epochs, default: 100")
    arg_parser.add_argument("--batch-size", type=int, default=32,
                            help="batch size for training, default: 32")
    arg_parser.add_argument("--lr", type=float, default=0.01, help="learning rate, default: 0.01")
    arg_parser.add_argument("--embed-size", type=int, default=25, help="embedding dimensions, default: 25")
    arg_parser.add_argument("--task-number", type=int, default=1,
                            help="task to process, default: 1")
    arg_parser.add_argument("--hops", type=int, default=1, help="Number of hops to make: 1, 2 or 3; default: 1 ")
    arg_parser.add_argument("--anneal-factor", type=int, default=2,
                            help="factor to anneal by every 'anneal-epoch(s)', default: 2")
    arg_parser.add_argument("--anneal-epoch", type=int, default=25,
                            help="anneal every [anneal-epoch] epoch, default: 25")
    arg_parser.add_argument("--eval", type=int, default=1, help="evaluate after training, default: 1")
    arg_parser.add_argument("--cuda", type=int, default=0, help="train on GPU, default: 0")
    arg_parser.add_argument("--memory-size", type=int, default=50, help="upper limit on memory size, default: 50")
    arg_parser.add_argument("--log-epochs", type=int, default=4,
                            help="Number of epochs after which to log progress, default: 4")
    arg_parser.add_argument("--joint-training", type=int, default=0, help="joint training flag, default: 0")

    arg_parser.add_argument("--saved-model-dir", type=str,
                            default="./saved/", help="path to folder where trained model will be saved.")
    arg_parser.add_argument("--data-dir", type=str, default="./data/tasks_1-20_v1-2/en",
                            help="path to folder from where data is loaded")

    arg_parser.add_argument("--debug", type=bool, default=False, help="Flag for debugging purposes")

    args = arg_parser.parse_args()

    check_paths(args)
    save_model_path = model_path(args)

    train_batches, val_batches, test_batches, train_set, val_set, test_set, sentence_size, vocab_size, story_size, word_idx = \
        process_data(args)

    if args.train == 1:
        train_network(train_batches, val_batches, test_batches, train_set, val_set, test_set, story_size=story_size,
                      vocab_size=vocab_size, save_model_path=save_model_path, args=args)

    if args.eval == 1:
        model = save_model_path
        eval_network(story_size=story_size, vocab_size=vocab_size,
                     EMBED_SIZE=args.embed_size, batch_size=args.batch_size, depth=args.hops,
                     model=model, test_batches=test_batches, test=test_set)


if __name__ == '__main__':
    main()
