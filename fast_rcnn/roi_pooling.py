import torch
import torch.nn as nn
import torch.autograd as ag
import math

from torch.autograd.function import Function
from torch._thnn import type2backend

class AdaptiveMaxPool2d(Function):
    def __init__(self, out_w, out_h):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input):
        output = input.new()
        indices = input.new().long()
        self.save_for_backward(input)
        self.indices = indices
        self._backend = type2backend[type(input)]
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        grad_input = grad_output.new()
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices)
        return grad_input, None

def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0],size[1])(input)

def roi_pooling(input, rois, size=(7,7), spatial_scale=1.0):
    assert(rois.dim() == 2)
    assert(rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    
    rois[:,1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)

if __name__ == '__main__':
    input = ag.Variable(torch.rand(1,1,10,10), requires_grad=True)
    rois = ag.Variable(torch.LongTensor([[0,1,2,7,8],[0,3,3,8,8]]),requires_grad=False)
    #rois = ag.Variable(torch.LongTensor([[0,3,3,8,8]]),requires_grad=False)

    out = adaptive_max_pool(input,(3,3))
    out.backward(out.data.clone().uniform_())

    out = roi_pooling(input, rois, size=(3,3))
    out.backward(out.data.clone().uniform_())

