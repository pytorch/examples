import torch.nn as nn
import numpy as np

# clean up environment
from bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes, filter_boxes

class FasterRCNN(nn.Container):

  def __init__(self):
    self.rpn_param = {
        '_feat_stride':16
    }

  # need to have a train and test mode
  # should it support batched images ?
  # need to pass target as argument only in train mode
  def forward(self, x):
    if self.train is True:
      im, gt = x
      # call model.train() here ?
    else
      im = x

    feats = self._features(im)

    # improve
    # it is used in get_anchors and also present in roi_pooling
    self._feat_stride = round(im.size(4)/feats.size(4))
    # rpn
    # put in a separate function
    rpn_map, rpn_bbox_transform = self._rpn_classifier(feats)
    all_anchors = self.rpn_get_anchors(im)
    #rpn_boxes = self.rpn_estimate(all_anchors, rpn_map)
    if self.train is True:
      rpn_labels, rpn_bbox_targets = self.rpn_targets(all_anchors, im, gt)
      # need to subsample boxes here
      rpn_loss = self.rpn_loss(rpn_map, rpn_bbox_transform, rpn_labels, rpn_bbox_targets)

    # roi proposal
    # clip, sort, pre nms topk, nms, after nms topk
    # proposal_layer.py
    # roi_boxes = self.get_roi_boxes(rpn_map, rpn_boxes)
    roi_boxes = self.get_roi_boxes(all_anchors, rpn_map, rpn_bbox_transform)

    if self.train is True:
      # append gt boxes and sample fg / bg boxes
      # proposal_target-layer.py
      roi_boxes, frcnn_labels, frcnn_bbox_targets = self.frcnn_targets(roi_boxes, im, gt)

    # r-cnn
    regions = self._roi_pooling(feats, roi_boxes)
    scores, bbox_transform = self._classifier(regions)

    boxes = self.bbox_reg(roi_boxes, bbox_transform)

    # apply cls + bbox reg loss here
    if self.train is True:
      frcnn_loss = self.frcnn_loss(scores, boxes, frcnn_labels, frcnn_bbox_targets)
      loss = frcnn_loss + rpn_loss
      return loss, scores, boxes

    return scores, boxes

  # the user define their model in here
  def _features(self, x):
    # _feat_stride should be defined / inferred from here
    pass
  def _classifier(self, x):
    pass
  def _roi_pooling(self, x):
    pass
  def _rpn_classifier(self, x):
    pass

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

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(anchors, gt_boxes)#.numpy()
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    
    # assign bg labels first so that positive labels can clobber them
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
      disable_inds = npr.choice(
          fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
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
  def get_roi_boxes(self, all_anchors, rpn_map, rpn_bbox_deltas)

    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    #scores = bottom[0].data[:, self._num_anchors:, :, :]
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = filter_boxes(proposals, min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
      order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
      keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    return roi_boxes
