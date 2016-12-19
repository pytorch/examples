import torch
import torch.nn as nn
import torch.autograd as ag
import math

# approximation for the adaptive max pooling which is currently missing from nn
# doesn't work if the input is smaller than size
def adaptive_max_pool(input, size):
    s = input.size()[2:]
    assert(s[0]>= size[0] and s[1] >= size[1])
    ratio = [float(x)/y for x,y in zip(s, size)]
    kernel_size = [int(math.ceil(x)) for x in ratio]
    stride = kernel_size
    remainder = [x*y-z for x, y, z in zip(kernel_size, size, s)]
    padding = [int(math.floor((x+1)/2)) for x in remainder]
    return nn.MaxPool2d(kernel_size,stride,padding=padding, ceil_mode=True)(input)
    #return nn.MaxPool2d(kernel_size,stride,padding=padding, ceil_mode=False)(input)

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
        im = input.narrow(0, im_idx, 1)[..., roi[2]:roi[4], roi[1]:roi[3]]
        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)

if __name__ == '__main__':
    input = ag.Variable(torch.rand(1,1,10,10), requires_grad=True)
    rois = ag.Variable(torch.LongTensor([[0,1,2,7,8],[0,3,3,8,8]]),requires_grad=False)
    #rois = ag.Variable(torch.LongTensor([[0,3,3,8,8]]),requires_grad=False)

    out = roi_pooling(input, rois, size=(3,3))
    out.backward(out.data.clone().uniform_())

