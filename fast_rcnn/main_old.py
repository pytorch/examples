import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.trainer as trainer
import torch.utils.data
import numpy as np

from roi_pooling import roi_pooling
from voc import VOCDetection, TransformVOCDetectionAnnotation

from tqdm import tqdm


cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip(cls, range(len(cls))))


train = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train',
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))


# TODO
# add class information in dataset
# separate in different files
# remove hard-coding 21 from Sampler
# cache the sampled boxes ?

# image flip goes to the dataset class, not BoxSampler

def bbox_overlaps(a, bb):
  #b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
  oo = []

  for b in bb:

    x1 = a.select(1,0).clone()
    x1[x1.lt(b[0])] = b[0] 
    y1 = a.select(1,1).clone()
    y1[y1.lt(b[1])] = b[1]
    x2 = a.select(1,2).clone()
    x2[x2.gt(b[2])] = b[2]
    y2 = a.select(1,3).clone()
    y2[y2.gt(b[3])] = b[3]

    w = x2-x1+1
    h = y2-y1+1
    inter = torch.mul(w,h).float()
    aarea = torch.mul((a.select(1,2)-a.select(1,0)+1), (a.select(1,3)-a.select(1,1)+1)).float()
    barea = (b[2]-b[0]+1) * (b[3]-b[1]+1)

    # intersection over union overlap
    o = torch.div(inter , (aarea+barea-inter))
    # set invalid entries to 0 overlap
    o[w.lt(0)] = 0
    o[h.lt(0)] = 0

    oo += [o]

  return torch.cat([o.view(-1,1) for o in oo],1)

class BoxGenerator(object):
    def __init__(self, num_boxes=2000):
        super(BoxGenerator, self).__init__()
        self.num_boxes = num_boxes

    def __call__(self, im):
        #h, w = im.size()[1:]
        w, h = im.size
        x = torch.LongTensor(self.num_boxes, 2).random_(0,w-1).sort(1)
        y = torch.LongTensor(self.num_boxes, 2).random_(0,h-1).sort(1)
    
        x = x[0]
        y = y[0]

        return torch.cat([x.select(1,0), y.select(1,0), x.select(1,1), y.select(1,1)], 1)


class BoxSampler(torch.utils.data.Dataset):

    def __init__(self, dataset, fg_threshold=0.5, bg_threshold=(0.0,0.5), 
            generate_boxes=BoxGenerator(num_boxes=10000)):
        super(BoxSampler, self).__init__()
        self.dataset = dataset
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.generate_boxes = generate_boxes

    def _overlap_and_attribute(self, boxes, gt_roidb):

        #overlaps = np.zeros((boxes.size(0), self.num_classes), dtype=np.float32)
        overlaps = np.zeros((boxes.size(0), 21), dtype=np.float32)

        if gt_roidb is not None and gt_roidb['boxes'].size > 0:
            gt_boxes = gt_roidb['boxes']
            gt_classes = np.array(gt_roidb['gt_classes'])
            gt_overlaps = bbox_overlaps(boxes,gt_boxes).numpy()
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)

            # remove low scoring
            pos = maxes >= self.fg_threshold
            neg = (maxes >= self.bg_threshold[0]) & (maxes < self.bg_threshold[1])
            maxes[neg] = 0

            I = np.where(maxes > 0)[0]
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = overlaps[pos | neg]
            boxes = boxes.numpy()
            boxes = boxes[pos | neg]
            return torch.from_numpy(boxes), torch.from_numpy(overlaps.argmax(axis=1))

    def __getitem__(self, idx):
        im, gt = self.dataset[idx]
        boxes = self.generate_boxes(im)
        boxes, labels = self._overlap_and_attribute(boxes, gt)

        if True:
            w, h = im.size
            im = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            im = im.view(h, w, 3)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            im = im.transpose(0, 1).transpose(0, 2).contiguous()
            im = im.float().div_(255)

        return im, boxes, labels

    def __len__(self):
        return len(self.dataset)


class BoxSelector(torch.utils.data.Dataset):
    def __init__(self, dataset, num_boxes=128, fg_fraction=0.25):
        super(BoxSelector, self).__init__()
        self.dataset = dataset
        self.num_boxes = num_boxes
        self.fg_fraction = fg_fraction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, boxes, labels = self.dataset[idx]

        boxes = boxes.numpy()
        labels = labels.numpy()

        bg = np.where(labels == 0)[0]
        fg = np.where(labels != 0)[0]
        nfg = min(len(fg), self.num_boxes*self.fg_fraction)
        nbg = min(len(bg), self.num_boxes - nfg)

        bg = bg[np.random.permutation(len(bg))[:nbg]]
        fg = fg[np.random.permutation(len(fg))[:nfg]]

        I = np.concatenate([fg, bg], axis=0)

        return im, torch.from_numpy(boxes[I]), torch.from_numpy(labels[I])


class ToPILImage(object):
    """ Converts a torch.*Tensor of range [0, 1] and shape C x H x W 
    or numpy ndarray of dtype=uint8, range[0, 255] and shape H x W x C
    to a PIL.Image of range [0, 255]
    """
    def __call__(self, pic):
        from PIL import Image, ImageOps
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = Image.fromarray(pic)
        else:
            npimg = pic.mul(255).byte().numpy()
            npimg = np.transpose(npimg, (1,2,0))
            img = Image.fromarray(npimg)
        return img

def make_grid(tensor, nrow=8, padding=2):
    import math
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W),
    or a list of images all of the same size,
    makes a grid of images
    """
    tensorlist = None
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = torch.Size(torch.Size([long(numImages)]) + tensorlist[0].size())
        tensor = tensorlist[0].new(size)
        for i in range(numImages):
            tensor[i].copy_(tensorlist[i])
    if tensor.dim() == 2: # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3: # single image
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1: # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps, width * xmaps).fill_(tensor.max())
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y*height+1+padding//2,height-padding)\
                .narrow(2, x*width+1+padding//2, width-padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid



ds = BoxSelector(BoxSampler(train, fg_threshold=0.75), 64, 0.25)

def collate_fn(batch):
    imgs, boxes, labels = zip(*batch)
    max_size = [max(size) for size in zip(*[im.size() for im in imgs])]
    new_imgs = imgs[0].new(len(imgs), *max_size).fill_(0)
    for im, im2 in zip(new_imgs, imgs):
        im.narrow(1,0,im2.size(1)).narrow(2,0,im2.size(2)).copy_(im2)
    boxes = np.concatenate([np.column_stack((np.full(t.size(0), i, dtype=np.int64), t.numpy())) for i, t in enumerate(boxes, 0)], axis=0)
    boxes = torch.from_numpy(boxes)
    labels = torch.cat(labels, 0)
    return new_imgs, boxes, labels

train_loader = torch.utils.data.DataLoader(
            ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)


def show(img, boxes, label, cls=None):
    from PIL import Image, ImageDraw
    #img, target = self.__getitem__(index)
    if cls is None:
        cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

    draw = ImageDraw.Draw(img)
    for obj, t in zip(boxes, label):
        if t > 0:
            draw.rectangle(obj[0:4].tolist(), outline=(255,0,0))
            draw.text(obj[0:2].tolist(), cls[t], fill=(0,255,0))
        else:
            #pass
            draw.rectangle(obj[0:4].tolist(), outline=(0,0,255))
    img.show()


for i, (img, boxes, labels) in tqdm(enumerate(train_loader)):
    #grid = make_grid(img, 2, 1)
    #grid = ToPILImage()(grid)
    #grid.show()
    #break
    pass
    #print('====')
    #print(i)
    #print(img.size())
    #print(boxes.size())
    #print(labels.size())

#im, box, label = ds[10]
#show(im,box,label)
