"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks

    2016 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
import math
from tqdm import tqdm
from nested_dict import nested_dict
from collections import OrderedDict
import torch
import torch.optim
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torch.legacy.nn import SpatialZeroPadding, SpatialReflectionPadding
import torchnet as tnt
from torchnet.engine import Engine
import torchnet.meter as meter
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--dropout', default=0, type=float)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)

# Device options
parser.add_argument('--save', default='/tmp/cifar', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training, 0 to for CPU')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# TODO: move this to main
opt = parser.parse_args()
print 'parsed options:', vars(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
epoch_step = json.loads(opt.epoch_step)

use_cuda = torch.cuda.is_available() and opt.ngpu > 0


class RandomCrop(object):
    """Random cropping with zero-padding or reflections
    note: have to use legacy nn here because there's no padding with reflection
    in PIL or skimage (could use OpenCV, but pain to install)
    """
    def __init__(self, pad=4, crop_type='zero'):
        if crop_type == 'zero':
            self.module = SpatialZeroPadding(pad, pad, pad, pad)
        elif crop_type == 'reflection':
            self.module = SpatialReflectionPadding(pad, pad, pad, pad)
        self.pad = pad

    def __call__(self, tensor):
        padded = self.module.forward(tensor.unsqueeze(0))[0]
        x = np.random.randint(0, self.pad*2)
        y = np.random.randint(0, self.pad*2)
        return padded[:, y:y+tensor.size(1), x:x+tensor.size(2)].contiguous()


def get_iterator(dataset, mode):
    """Creates iterator for train (mode=True) or test (mode=False) subset
    """
    mean = [125.3, 123.0, 113.9]
    std = [63.0,  62.1,  66.7]

    randomflip = lambda x: np.fliplr(x) if np.random.random() > 0.5 else x
    normalize = lambda x: (x.astype(np.float32) - mean) / std
    totensor = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float()

    if mode:
        transforms = [randomflip, normalize, totensor,
                      RandomCrop(4, 'reflection')]
    else:
        transforms = [normalize, totensor]

    dataset_constr = getattr(datasets, dataset)
    ds = dataset_constr('.', train=mode, download=True)
    data = ds.train_data if mode else ds.test_data
    labels = ds.train_labels if mode else ds.test_labels
    ds_tnt = tnt.dataset.TensorDataset([data.transpose(0, 2, 3, 1), labels])
    ds_tnt = ds_tnt.transform({0: tnt.transform.compose(transforms)})
    return ds_tnt.parallel(batch_size=opt.batchSize, shuffle=mode,
                           num_workers=opt.nthread)


def cast(params, dtype='float'):
    """
    :param params: tensor of dict of tensors
    :param dtype: 'float', 'double' or 'half'
    :return: out-of-place cast
    """
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if use_cuda else params, dtype)()


def resnet(depth, width, num_classes, dropout):
    """Wide ResNet model definition

    :param depth: total number of layers
    :param width: multiplier for number of feature planes in each convolution
    :param num_classes: number of output neurons in the top linear layer
    :return:
      f: function that defines the model
      params: optimizable parameters dict
      stats: batch normalization moments
    """
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = int((depth - 4) / 6)
    widths = np.floor(np.asarray([16., 32., 64.]) * width).astype(np.int)

    def conv_params(ni, no, k=1):
        """MSRA init for convolutional layers (biasless).

        :param ni: number of input feature planes
        :param no: number of output feature planes
        :param k: kernel width (assumed square)
        :return: filters tensor
        """
        return cast(torch.Tensor(no, ni, k, k).normal_(0, 2/math.sqrt(ni*k*k)))

    def linear_params(ni, no):
        """MSRA init for the last linear layer.

        :param ni: number of input feature planes
        :param no: number of output feature planes
        :return: dict with weights and biases
        """
        return cast(dict(
            weight=torch.Tensor(no, ni).normal_(0, 2/math.sqrt(ni)),
            bias=torch.zeros(no)))

    def bnparams(ni):
        """Batch normalization learnable parameters."""
        return cast(dict(
            weight=torch.Tensor(ni).uniform_(),
            bias=torch.zeros(ni)))

    def bnstats(ni):
        """Batch normalization moments."""
        return cast(dict(
            running_mean=torch.zeros(ni),
            running_var=torch.ones(ni)))

    def gen_block_params(ni, no):
        return {
                'conv0': conv_params(ni, no, 3),
                'conv1': conv_params(no, no, 3),
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'convdim': conv_params(ni, no, 1) if ni != no else None,
                }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no),
                                'bn1': bnstats(no)}
                for i in range(count)}

    params = nested_dict({
            'conv0': conv_params(3, 16, 3),
            'group0': gen_group_params(16, widths[0], n),
            'group1': gen_group_params(widths[0], widths[1], n),
            'group2': gen_group_params(widths[1], widths[2], n),
            'bn': bnparams(widths[2]),
            'fc': linear_params(widths[2], num_classes),
            })

    stats = nested_dict({
            'group0': gen_group_stats(16, widths[0], n),
            'group1': gen_group_stats(widths[0], widths[1], n),
            'group2': gen_group_stats(widths[1], widths[2], n),
            'bn': bnstats(widths[2]),
            })

    flat_params = OrderedDict()
    flat_stats = OrderedDict()
    for keys, v in params.iteritems_flat():
        if v is not None:
            flat_params['.'.join(keys)] = Variable(v, requires_grad=True)
    for keys, v in stats.iteritems_flat():
        flat_stats['.'.join(keys)] = v

    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight=params[base+'.weight'],
                                   bias=params[base+'.bias'],
                                   running_mean=stats[base+'.running_mean'],
                                   running_var=stats[base+'.running_var'],
                                   training=mode, momentum=0.1, eps=1e-5))

    def block(x, params, stats, base, mode, stride):
        o1 = activation(x, params, stats, base+'.bn0', mode)
        y = F.conv2d(o1, params[base+'.conv0'], stride=stride, padding=1)
        o2 = activation(y, params, stats, base+'.bn1', mode)
        if opt.dropout > 0:
            o2 = F.dropout(o2, p=0.5, training=mode)
        z = F.conv2d(o2, params[base+'.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base+'.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base, i), mode,
                      stride if i == 0 else 1)
        return o

    def f(inputs, params, stats, mode):
        x = F.conv2d(inputs, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = activation(g2, params, stats, 'bn', mode)
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params, flat_stats


def data_parallel(f, inputs, params, stats, mode, device_ids,
                  output_device=None):
    """Functional data parallel implementation.

    Given an input tensor splits it into len(device_ids) parts and applies `f`
    to each subtensor, and concatenates outputs

    :param f: a function
    :param inputs: Variable containing inputs
    :param params: dict of optimizable Variables
    :param stats: dict of moments tensors
    :param mode: training or not
    :param device_ids: list of GPU ids to use, e.g. [0,1] to train on 2 cards,
      or [] to train on the first one if CPU
    :param output_device: where to put the outputs
    :return: outputs tensor
    """
    if len(device_ids) == 1 or len(device_ids) == 0:
        return f(inputs, params, stats, mode)

    if output_device is None:
        output_device = device_ids[0]

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k, v in param_dict.iteritems():
            for i, u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [lambda x, p=p, s=s, mode=mode: f(x, p, s, mode)
                for i,(p,s) in enumerate(zip(params_replicas, stats_replicas))]
    inputs = scatter(inputs, device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def main():
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100
    iter_train = get_iterator(opt.dataset, True)
    iter_test = get_iterator(opt.dataset, False)

    f, params, stats = resnet(opt.depth, opt.width, num_classes, opt.dropout)

    optimizer = torch.optim.SGD(params.values(), opt.lr, 0.9, opt.weightDecay)

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params, stats = state_dict['params'], state_dict['stats']
        optimizer = state_dict['optimizer']

    n_parameters = sum([p.numel() for p in params.values() + stats.values()])
    print '\nTotal number of parameters:', n_parameters

    meter_loss = meter.AverageValueMeter()
    classacc = meter.ClassErrorMeter(accuracy=True)
    timer_train = meter.TimeMeter('s')
    timer_test = meter.TimeMeter('s')

    def h(sample):
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        mode = sample[2]
        y = data_parallel(f, inputs, params, stats, mode, np.arange(opt.ngpu))
        return F.cross_entropy(y, targets), y

    def log(t):
        torch.save(dict(params=params, stats=stats,
                        optimizer=optimizer, epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'w'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as ff:
            ff.write('json_stats: ' + json.dumps(z) + '\n')
        print z

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(iter_train)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            for group in state['optimizer'].param_groups:
                group['lr'] *= opt.lr_decay_ratio
                for p in group['params']:
                    param_state = state['optimizer'].state[id(p)]
                    param_state['momentum_buffer'].zero_()

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, iter_test)

        print log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": classacc.value()[0],
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
           })

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, iter_train, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
