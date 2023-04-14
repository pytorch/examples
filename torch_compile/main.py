from __future__ import print_function
import argparse
import logging
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import torch._dynamo.config as dcfg
import torch._functorch as fcfg
import torch._inductor.config as icfg

from datasets import load_dataset
import matplotlib.pyplot as plt
from statistics import median

torch.set_float32_matmul_precision('high')
cudnn.benchmark=True

##################################################
#               DEBUG FLAGS
##################################################
#TORCH_COMPILE_DEBUG=1
#dcfg.log_level = logging.DEBUG
#dcfg.verbose = True
#dcfg.print_graph_breaks = True
#dcfg.output_code = True
#AOT_FX_GRAPHS=1
#fcfg.log_level = logging.DEBUG
#fcfg.debug_graphs = True
#icfg.debug = True
#icfg.trace.enabled = True
##################################################

def train(args, model, device, train_loader, optimizer, criterion, epoch, profile):
    """
    Train the model
    """
    model.train()

    if profile:
        end = time()
        forward_duration_ms, backward_duration_ms, data_loading_duration_ms = [], [], []
    for batch_idx, batch in enumerate(train_loader):

        if profile:
            data_loading_duration_ms.append((time() - end) * 1e3)

            torch.cuda.synchronize()
            pre_forward_time = time()

        data, target = batch["image"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if profile:
            torch.cuda.synchronize()
            post_forward_time = time()

        loss.backward()

        if profile:
            torch.cuda.synchronize()
            post_backward_time = time()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        
        if profile:
            forward_duration_ms.append((post_forward_time - pre_forward_time) * 1e3)
            backward_duration_ms.append((post_backward_time - post_forward_time) * 1e3)
            end = time()

            if batch_idx % args.log_interval == 0:
                print("Median forward time (ms) {:.2f} | backward time (ms) {:.2f} | dataloader time (ms) {:.2f}".format(
                    median(forward_duration_ms), median(backward_duration_ms), median(data_loading_duration_ms)
                ))
    if profile:    
        print("Total forward time (s) {:.2f} | backward time (s) {:.2f} | dataloader time (s) {:.2f}".format(
            sum(forward_duration_ms)/1000, sum(backward_duration_ms)/1000, sum(data_loading_duration_ms)/1000
        ))

def test(model, device, test_loader, criterion):
    """
    Evaluate the model
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch["image"].to(device), batch["labels"].to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def timed(fn):
    """
    Method to measure the execution time of a function
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--torch-compile', action='store_true', default=False,
                        help='To enable torch.compile')
    parser.add_argument('--reduce-overhead', action='store_true', default=False,
                        help='To enable reduce-overhead mode in torch.compile')
    parser.add_argument('--max-autotune', action='store_true', default=False,
                        help='To enable max-autotune mode in torch.compile')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD Momentum (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dl-opt', action='store_true', default=False,
                        help='Dataloader optimization')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='For detailed profiling')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plotting training time over epoch')
    parser.add_argument('--resnet152', action='store_true', default=False,
                        help='Use ResNet152 model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    assert use_cuda == True, "This tutorial needs a CUDA device"
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    # DataLoader Configuration
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if args.dl_opt:
        opt_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                     }
        train_kwargs.update(opt_kwargs)
        test_kwargs.update(opt_kwargs)

    # Transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    # Transformation function for loading HF dataset into torch dataloader
    def train_transform_func(examples):
        examples["image"] =  [train_transform(pil_img.convert("RGB")) for pil_img in examples["image"]]
        return examples
    def test_transform_func(examples):
        examples["image"] =  [val_transform(pil_img.convert("RGB")) for pil_img in examples["image"]]
        return examples
    ds = load_dataset("cats_vs_dogs", split="train")
    ds = ds.train_test_split(test_size=0.2, shuffle=True)
    train_ds, test_ds = ds["train"], ds["test"]
    train_ds.set_transform(train_transform_func)
    test_ds.set_transform(test_transform_func)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_ds,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    if not args.resnet152:
        print("Using ResNet18")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        print("Using ResNet152")
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    # We have 2 classes in the dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)

    # Configuration for torh.compile
    if args.torch_compile:
        print("torch.compile() enabled")
        if args.reduce_overhead:
            print("Mode 'reduce-overhead' enabled")
            opt_model = torch.compile(model, mode="reduce-overhead")
        elif args.max_autotune:
            print("Mode 'max-autotune' enabled")
            opt_model = torch.compile(model, mode="max-autotune")
        else:
            opt_model = torch.compile(model)
    else:
        print("torch.compile() disabled")
        opt_model = model
    
    optimizer = optim.SGD(opt_model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    
    total_training_time, total_evaluation_time = [], []
    epochs = []
    for epoch in range(1, args.epochs + 1):
        training_time = timed(lambda: train(args, opt_model, device, train_loader, optimizer, criterion, epoch, args.profile))[1]
        print(f"Training Time: {training_time}")
        total_training_time.append(training_time)
        epochs.append(epoch)
             
        evautation_time = timed(lambda: test(opt_model, device, test_loader, criterion))[1]
        print(f"Evaluation Time: {evautation_time}")
        total_evaluation_time.append(evautation_time)
        scheduler.step()
        print("#########################################################")

    
    print("#########################################################")
    print(f"Total training time in seconds: {sum(total_training_time):.2f}")
    print(f"Total evaluation time in seconds: {sum(total_evaluation_time):.2f}")
    print("#########################################################")
    if args.save_model:
        torch.save(opt_model.state_dict(), "resnet_cats_dogs.pt")
    
    if args.plot:
        fig = "training_time.png"
        print(f"Saving plot of training time in {fig}")
        plt.plot(epochs, total_training_time)
        plt.title('Training Time of ResNet18 with torch.compile on A10 GPU')
        plt.xlabel('Epochs')
        plt.ylabel('Training Time')
        plt.savefig(fig)


if __name__ == '__main__':
    main()
