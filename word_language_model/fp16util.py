import torch


def params_to_type(params, totype):
    new_params = []
    for param in params:
        new_params.append(param.type(totype))
    return new_params


def params_to_16(params):
    return params_to_type(params, torch.cuda.HalfTensor)


def params_to_32(params):
    return params_to_type(params, torch.cuda.FloatTensor)


def clone_params(net):
    new_params = []
    for param in list(net.parameters()):
        new_params.append(param.data.clone())
    return new_params


def clone_grads(net):
    new_params = []
    for param in list(net.parameters()):
        new_params.append(param.grad.data.clone())
    return new_params


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i])
