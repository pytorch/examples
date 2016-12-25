import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

# clean up environment
from utils import bbox_transform, bbox_transform_inv, clip_boxes, filter_boxes, bbox_overlaps
from generate_anchors import generate_anchors

from utils import to_var as _tovar

from py_cpu_nms import py_cpu_nms as nms

class RPN(nn.Container):

  def __init__(self,
      classifier, anchor_scales=None,
      negative_overlap=0.3, positive_overlap=0.7,
      fg_fraction=0.5, batch_size=256,
      nms_thresh=0.7, min_size=16,
      pre_nms_topN=12000, post_nms_topN=2000
      ):
    super(RPN, self).__init__()

    self.rpn_classifier = classifier

    if anchor_scales is None:
      anchor_scales = (8, 16, 32)
    self._anchors = generate_anchors(scales=np.array(anchor_scales))
    self._num_anchors = self._anchors.shape[0]

    self.negative_overlap = negative_overlap
    self.positive_overlap = positive_overlap
    self.fg_fraction = fg_fraction
    self.batch_size = batch_size

    # used for both train and test
    self.nms_thresh = nms_thresh
    self.pre_nms_topN = pre_nms_topN
    self.post_nms_topN = post_nms_topN
    self.min_size = min_size


  # output rpn probs as well
  def forward(self, im, feats, gt=None):
    assert im.size(0) == 1, 'only single element batches supported'
    # improve
    # it is used in get_anchors and also present in roi_pooling
    self._feat_stride = round(im.size(3)/feats.size(3))
    # rpn
    # put in a separate function
    rpn_map, rpn_bbox_pred = self.rpn_classifier(feats)
    all_anchors = self.rpn_get_anchors(feats)
    rpn_loss = None
    #if self.training is True:
    if gt is not None:
      assert gt is not None
      rpn_labels, rpn_bbox_targets = self.rpn_targets(all_anchors, im, gt)
      # need to subsample boxes here
      rpn_loss = self.rpn_loss(rpn_map, rpn_bbox_pred, rpn_labels, rpn_bbox_targets)

    # roi proposal
    # clip, sort, pre nms topk, nms, after nms topk
    # params are different for train and test
    # proposal_layer.py
    roi_boxes, scores = self.get_roi_boxes(all_anchors, rpn_map, rpn_bbox_pred, im)
    # only for visualization
    if False:
      roi_boxes = all_anchors
      return _tovar((roi_boxes, scores, rpn_loss, rpn_labels))

    return _tovar((roi_boxes, scores, rpn_loss))


  # from faster rcnn py
  def rpn_get_anchors(self, im):
    height, width = im.size()[-2:]
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * self._feat_stride
    shift_y = np.arange(0, height) * self._feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = self._num_anchors
    K = shifts.shape[0]
    all_anchors = (self._anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors

  # restructure because we don't want -1 in labels
  # shouldn't we instead keep only the bboxes for which labels >= 0?
  def rpn_targets(self, all_anchors, im, gt):
    total_anchors = all_anchors.shape[0]
    gt_boxes = gt['boxes']

    height, width = im.size()[-2:]
    # only keep anchors inside the image
    _allowed_border = 0
    inds_inside = np.where(
         (all_anchors[:, 0] >= -_allowed_border) &
         (all_anchors[:, 1] >= -_allowed_border) &
         (all_anchors[:, 2] < width  + _allowed_border) &  # width
         (all_anchors[:, 3] < height + _allowed_border)    # height
    )[0]
     
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    assert anchors.shape[0] > 0, '{0}x{1} -> {2}'.format(height,width,total_anchors)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    #overlaps = bbox_overlaps(anchors, gt_boxes)#.numpy()
    overlaps = bbox_overlaps(torch.from_numpy(anchors), gt_boxes).numpy()
    gt_boxes = gt_boxes.numpy()
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    
    # assign bg labels first so that positive labels can clobber them
    labels[max_overlaps < self.negative_overlap] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= self.positive_overlap] = 1
    
    # subsample positive labels if we have too many
    num_fg = int(self.fg_fraction * self.batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
      disable_inds = npr.choice(
          fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = self.batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
      disable_inds = npr.choice(
          bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      labels[disable_inds] = -1

    #bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    return labels, bbox_targets

  # I need to know the original image size (or have the scaling factor)
  def get_roi_boxes(self, anchors, rpn_map, rpn_bbox_deltas, im):
    # TODO fix this!!!
    im_info = (100, 100, 1)

    bbox_deltas = rpn_bbox_deltas.data.numpy()
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    #scores = bottom[0].data[:, self._num_anchors:, :, :]
    scores = rpn_map.data[:, self._num_anchors:, :, :].numpy()
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im.size()[-2:])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = filter_boxes(proposals, self.min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if self.pre_nms_topN > 0:
      order = order[:self.pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), self.nms_thresh)
    if self.post_nms_topN > 0:
      keep = keep[:self.post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    return proposals, scores

  def rpn_loss(self, rpn_map, rpn_bbox_transform, rpn_labels, rpn_bbox_targets):
    height, width = rpn_map.size()[-2:]

    rpn_map = rpn_map.view(-1, 2, height, width).permute(0,2,3,1).contiguous().view(-1, 2)
    labels = torch.from_numpy(rpn_labels).long() # convert properly
    labels = labels.view(1, height, width, -1).permute(0, 3, 1, 2).contiguous()
    labels = labels.view(-1)
  
    idx = labels.ge(0).nonzero()[:,0]
    rpn_map = rpn_map.index_select(0, Variable(idx, requires_grad=False))
    labels = labels.index_select(0, idx)
    labels = Variable(labels, requires_grad=False)
    
    rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets)
    rpn_bbox_targets = rpn_bbox_targets.view(1, height, width, -1).permute(0, 3, 1, 2)
    rpn_bbox_targets = Variable(rpn_bbox_targets, requires_grad=False)

    cls_crit = nn.CrossEntropyLoss()
    reg_crit = nn.SmoothL1Loss()
    cls_loss = cls_crit(rpn_map, labels)
    # verify normalization and sigma
    reg_loss = reg_crit(rpn_bbox_transform, rpn_bbox_targets)

    loss = cls_loss + reg_loss
    return loss

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def show(img, boxes, label):
    from PIL import Image, ImageDraw
    import torchvision.transforms as transforms
    #img, target = self.__getitem__(index)
    img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for obj, t in zip(boxes, label):
        #print(type(t))
        if t == 1:
            #print(t)
            draw.rectangle(obj[0:4].tolist(), outline=(255,0,0))
            #draw.text(obj[0:2].tolist(), cls[t], fill=(0,255,0))
        #else:
        elif t == 0:
            #pass
            draw.rectangle(obj[0:4].tolist(), outline=(0,0,255))
    img.show()



if __name__ == '__main__':
  import torch
  from voc import VOCDetection, TransformVOCDetectionAnnotation
  import torchvision.transforms as transforms

  class RPNClassifier(nn.Container):
    def __init__(self, n):
      super(RPNClassifier, self).__init__()
      self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
      self.m2 = nn.Conv2d(n, 36, 3, 1, 1)

    def forward(self, x):
      return self.m1(x), self.m2(x)


  rpn = RPN(RPNClassifier(3))
  cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
  class_to_ind = dict(zip(cls, range(len(cls))))


  train = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train',
            transform=transforms.ToTensor(),
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))
  
  im, gt = train[100]
  im0 = im

  im = im.unsqueeze(0)

  feats = Variable(torch.rand(1,3,im.size(2)/16, im.size(3)/16))
  print(feats.size())
  print(im.size())

  #rpn.eval()
  rpn.train()
  import time
  t = time.time()
  #boxes, scores, loss, labels = rpn(im, feats, gt)
  boxes, scores, loss = rpn(im, feats, gt)
  print time.time() - t
  print loss
  loss.backward()

  show(im0, boxes.data, labels.data.int().tolist())

  #from IPython import embed; embed()
