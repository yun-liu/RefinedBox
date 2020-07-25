import caffe
import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        cfg_key = { 0: 'TRAIN', 1: 'TEST' }
        self.pre_nms_topN  = cfg[cfg_key[self.phase]].RERANK_PRE_NMS_TOP_N
        self.post_nms_topN = cfg[cfg_key[self.phase]].RERANK_POST_NMS_TOP_N
        self.nms_thresh    = cfg[cfg_key[self.phase]].RERANK_NMS_THRESH
        self.min_size      = cfg[cfg_key[self.phase]].RERANK_MIN_SIZE

        # rois blob: holds R regions of interest, each is a 4-tuple
        # (x1, y1, x2, y2) specifying a rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 4)

    def forward(self, bottom, top):
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        rois = bottom[0].data
        im_info = bottom[1].data
        scores = bottom[2].data[:, 1]
        bbox_deltas = bottom[3].data[:, 4:]
        levels = rois[:, 0]
        assert (levels == 0).all(), "Only single-scale batch implemented"

        rois = bbox_transform_inv(rois[:, 1:], bbox_deltas)
        rois = clip_boxes(rois, im_info[0, :2])

        keep = _filter_boxes(rois, self.min_size * im_info[0, 2])
        rois = rois[keep, :]
        scores = scores[keep]
        levels = levels[keep]

        order = scores.argsort()[::-1]
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        rois = rois[order, :]
        scores = scores[order]
        levels = levels[order]

        keep = nms(np.hstack((rois, scores[:, np.newaxis])), self.nms_thresh)
        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        rois = rois[keep, :]
        levels = levels[keep]
        rois = np.hstack((levels[:, np.newaxis], rois))

        top[0].reshape(*(rois.shape))
        top[0].data[...] = rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
