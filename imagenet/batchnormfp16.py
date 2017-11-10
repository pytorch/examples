import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable

class BatchNormFN(Function):

    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, momentum, eps):
        use_cudnn = (cudnn.is_acceptable(input) and
                     cudnn.version() > 5110 and
                     weight is not None and bias is not None)
        if not use_cudnn:
            raise RuntimeError("this module should be used only with cudnn")

        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        
        output = input.new(input.size())
        num_features = input.size(1)
        if torch.typename(input)=='torch.cuda.HalfTensor':
            _save_mean = torch.cuda.FloatTensor(num_features)
            _save_std = torch.cuda.FloatTensor(num_features)
        else:
            _save_mean = input.new(num_features)
            _save_std = input.new(num_features)

        torch._C._cudnn_batch_norm_forward(
            input, output, weight, bias,
            running_mean, running_var, _save_mean, _save_std,
            training, momentum, eps)

        ctx._save_mean = _save_mean
        ctx._save_std = _save_std
        ctx.running_mean = running_mean
        ctx.running_var = running_var

        ctx.save_for_backward(input, weight, bias)

        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        if not ctx.training:
            raise RuntimeError("This module does not support backwards in inference.")
        
        grad_input, grad_weight, grad_bias = None, None, None
        
        grad_input = input.new(input.size())
        grad_weight = weight.new(weight.size()).zero_()
        grad_bias = bias.new(bias.size()).zero_()

        torch._C._cudnn_batch_norm_backward(
            input, grad_output, grad_input,
            grad_weight, grad_bias, weight,
            ctx.running_mean, ctx.running_var,
            ctx._save_mean, ctx._save_std, ctx.training, ctx.eps)
        
        return grad_input, None, None, grad_weight, grad_bias, None, None, None

    
class BatchNorm2dFP16(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm2dFP16, self).__init__(num_features, eps, momentum, affine)
        if not self.affine:
            raise RuntimeError("This module does not support non-affine BN.")
        self.reset_parameters()
 
    def __repr__(self):
        return ('fp16{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))
    
    def forward(self,input):
        if torch.typename(input.data) == 'torch.cuda.HalfTensor':
            self.float()
        
        return BatchNormFN.apply(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)


