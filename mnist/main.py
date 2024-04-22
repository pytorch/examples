from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# model 是作为参数传入 train 函数的一个对象。根据代码上下文，model 应当是一个已经定义好的深度学习模型，它可以是：
# 现成的基础模型：这意味着 model 可能是基于某个预训练的深度学习基础模型构建的。例如，它可能是基于 PyTorch 中的
# torchvision.models 模块加载的预训练卷积神经网络（如 ResNet、VGG、DenseNet 等），或者基于 transformers 库加载的预训练自然语言处理模型（如 BERT、GPT 等）。
# 这样的模型通常会在特定任务上进行微调（fine-tuning），即在已有模型的基础上添加或替换部分层，然后使用新的训练数据继续训练整个模型。
#
# 自定义模型：model 可能是用户根据研究或应用需求从头设计并实现的深度学习模型。这种情况下，模型架构完全由用户定义，
# 可能包括卷积层、全连接层、循环层、注意力机制等各类神经网络组件，以及可能存在的批次归一化、dropout 等正则化技术。
#
# 无论是现成的基础模型还是自定义模型，当 model 传入 train 函数时，它应当已经完成了以下步骤：
# 模型定义：通过继承 torch.nn.Module 类并编写前向传播（forward）逻辑来定义模型结构。
# 模型初始化：创建模型实例，并可能设置了初始权重（如果是从头训练）或加载了预训练权重（如果是基于预训练模型）。
#
# train 函数的作用是针对传入的 model 进行训练，即在给定的训练数据、优化器、设备等条件下，通过迭代执行前向传播、反向传播、梯度更新等步骤，
# 使模型的权重参数逐渐适应训练数据分布，从而提升模型在特定任务上的性能。经过 train 函数训练后的 model，其内部权重参数将发生改变，
# 反映出了对训练数据的学习结果。因此，可以认为 train 函数的执行结果是一个经过训练（即权重已更新）的模型。如果您指的是最终得到的、
# 经过完整训练流程后的 model，那么答案是肯定的：train 函数处理后的 model 就是一个经过训练的模型。

def train(args, model, device, train_loader, optimizer, epoch):
    """
    对给定模型进行单个训练周期的训练。

    参数：
    - args (argparse.Namespace 或 dict-like 类型): 存储多种训练参数（如 log_interval 和 dry_run）的容器对象。
    - model (torch.nn.Module): 待训练模型，为 PyTorch 神经网络模块实例。
    - device (str): 指定模型与数据将在哪个设备（CPU 或 GPU）上运行。
    - train_loader (torch.utils.data.DataLoader): 提供训练数据的数据加载器，返回每批输入特征（`data`）和相应标签（`target`）。
    - optimizer (torch.optim.Optimizer): 用于根据计算出的梯度更新模型权重的优化器。
    - epoch (int): 当前训练周期数，用于日志记录。

    步骤：
    1. 将模型设置为训练模式。
    2. 遍历训练数据集中的批次。
    3. 将批次数据和标签移动至指定设备。
    4. 清除优化器中累积的梯度。
    5. 执行模型前向传播以获取预测结果。
    6. 计算预测与真实标签间的负对数似然损失（NLL Loss）。
    7. 反向传播损失以计算模型参数相对于损失的梯度。
    8. 使用优化器更新模型权重。
    9. （可选）按照 log_interval 定义的间隔打印训练进度与当前损失。
    10. （干运行模式）若开启，打印首个进度消息后终止训练循环。
    """

    model.train()  # 设置模型为训练模式

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将批次数据和标签移动至指定设备

        optimizer.zero_grad()  # 清除优化器中累积的梯度

        output = model(data)  # 执行模型前向传播以获取预测结果
        loss = F.nll_loss(output, target)  # 计算预测与真实标签间的负对数似然损失（NLL Loss）

        loss.backward()  # 反向传播损失以计算模型参数相对于损失的梯度
        optimizer.step()  # 使用优化器更新模型权重

        if batch_idx % args.log_interval == 0:  # 按照 log_interval 定义的间隔打印训练进度与当前损失
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if args.dry_run:  # 干运行模式：打印首个进度消息后终止训练循环
                break



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    """
    主函数，用于设置训练参数，并执行MNIST数据集的训练和测试。

    使用argparse解析命令行参数，配置训练过程的各种参数，包括批处理大小、学习率、训练轮数等。
    根据是否具备CUDA或macOS GPU环境，选择合适的设备进行训练。
    在训练过程中，每个epoch后进行测试，并可选择保存当前模型。
    """
    # 解析训练设置参数
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 根据命令行参数决定是否使用CUDA或macOS的GPU训练
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 选择训练使用的设备
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 打印训练使用的设备
    print("Using Device: {}\n".format(device))

    # 定义训练和测试的批处理参数，并根据是否使用CUDA添加额外的参数
    # 定义并初始化字典类型的训练参数

    # pin_memory: 主要用于PyTorch 的 DataLoader 类，主要生效于数据加载过程中。特别是在 CPU 主机内存向 GPU 显存传输数据时候，能够提升数据转移的效率。
    # 作用与目的： pin_memory=True 指令告诉 DataLoader 在预处理数据后，将数据张量（通常是 torch.Tensor 对象）固定（pin）在 CPU 的页锁定内存（Page-Locked Memory）中。

    # 页锁定内存是一种特殊的内存区域，其特点是：
    # 直接访问：GPU可以直接访问和操作这类内存中的数据，无需通过CPU中介，从而减少数据传输的延迟和CPU-GPU通信开销。
    # 不可交换：操作系统不会将其换出到磁盘，保证了数据在GPU需要时始终驻留在物理内存中，避免了数据传输过程中可能出现的页面失效（page fault）。
    #
    # 启用 pin_memory 的主要好处在于：
    # 提升数据传输效率：当数据位于页锁定内存时，从CPU到GPU的内存复制操作（如使用 to(device) 方法将数据转移到GPU上）可以利用零拷贝技术（Zero-copy），
    # 即通过DMA（Direct Memory Access）直接在硬件层面完成数据搬运，无需CPU参与，显著加快数据传输速度。
    # 优化多流并行性：在多GPU或多流（stream）场景下，页锁定内存有助于并发数据传输，使得CPU可以继续进行其他计算任务，同时GPU可以从页锁定内存中异步加载数据，提高了系统的整体利用率和训练速度。

    # 生效位置： pin_memory 参数生效的具体位置包括：
    # 数据加载循环：在 DataLoader 内部的迭代过程中，当一个批次的数据被预处理完毕后，若 pin_memory=True，则会调用 torch.Tensor.pin_memory() 方法将数据张量固定到页锁定内存。
    # 数据转移到GPU：当用户在训练循环中显式调用 data.to(device)（其中 device='cuda:0' 或其他GPU设备标识）将数据从CPU转移到GPU时，
    # 如果数据已经在页锁定内存中，那么可以利用高效的直接访问机制进行传输。

    # 总结起来，pin_memory 参数在 PyTorch 的 DataLoader 中启用后，主要在数据预处理完成后固定数据到页锁定内存，以及后续数据从CPU到GPU的转移过程中发挥作用，旨在提高数据加载和传输的效率，
    # 尤其在使用GPU进行深度学习训练时具有重要意义。
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据集的加载和转换
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 模型的加载和优化器的选择
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # 学习率调整策略
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 开始训练和测试
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # 如果设置了保存模型，则保存
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")



if __name__ == '__main__':
    main()

# 在您提供的Python代码片段中，确实使用了神经网络进行训练。具体体现在以下几点：
#
# 定义模型：虽然代码中没有直接展示Net类的定义，但model = Net().to(device)这一行表明创建了一个名为Net的神经网络实例，并将其参数移动到所选的设备（CPU、CUDA或macOS GPU）上。
# Net类应包含了多层神经元及其连接组成的网络结构，如全连接层、卷积层、池化层等，以及适当的激活函数。
#
# 数据预处理与加载：代码使用datasets.MNIST加载MNIST手写数字数据集，这是一个广泛用于训练神经网络的经典数据集。数据经过transforms.Compose定义的一系列转换，
# 包括将图像数据转换为张量（transforms.ToTensor()），以及进行标准化（transforms.Normalize），这些预处理步骤有助于神经网络更好地学习和泛化。
#
# 优化器与学习率调整：optimizer = optim.Adadelta(model.parameters(), lr=args.lr)创建了一个优化器（这里使用的是 Adadelta），
# 用于更新神经网络的权重和偏置以最小化损失函数。scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)定义了学习率调整策略，
# 即每过一个epoch（step_size=1），学习率按gamma比例衰减，这是神经网络训练中常见的调整学习速率的方法，旨在保持良好的收敛性能。
#
# 训练循环：for epoch in range(1, args.epochs + 1):开始训练循环，每次循环代表一个完整的数据集遍历（epoch）。
# 在每个epoch内，调用train(args, model, device, train_loader, optimizer, epoch)进行模型训练，使用test(model, device, test_loader)进行模型验证。
# 这两个函数内部应实现了前向传播、反向传播、梯度更新等神经网络训练的核心步骤。
#
# 综上所述，这段Python代码确实利用了神经网络进行MNIST数据集的训练，包括模型定义、数据预处理、优化器设置、学习率调度以及实际的训练与测试循环。尽管具体的神经网络架构（即Net类的定义）未在提供的代码片段中展示，但根据代码逻辑可以确定这是一个基于神经网络的训练流程。
