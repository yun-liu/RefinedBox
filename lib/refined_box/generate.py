from fast_rcnn.config import cfg
from utils.blob import im_list_to_blob
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import cv2

def _get_image_blob(im, im_rois):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[0]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    rois = im_rois * im_scale
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    rois = np.hstack((levels, rois))

    return blob, rois, im_info

def im_proposals(net, blobs):
    """Generate RPN proposals on a single image."""
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(
            data=blobs['data'].astype(np.float32, copy=False),
            rois=blobs['rois'].astype(np.float32, copy=False),
            im_info=blobs['im_info'].astype(np.float32, copy=False))

    scale = blobs['im_info'][0, 2]
    boxes = blobs_out['rerank_rois'][:, 1:].copy() / scale
    return boxes

def imdb_proposals(net, imdb):
    """Generate RPN proposals on all images in an imdb."""
    print 'Loading data ...'
    roidb = imdb.roidb

    _t = Timer()
    imdb_boxes = [[] for _ in xrange(imdb.num_images)]
    for i in xrange(imdb.num_images):
        blob = {}
        im = cv2.imread(imdb.image_path_at(i))
        rois = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
        blob['data'], blob['rois'], blob['im_info'] = _get_image_blob(im, rois)

        _t.tic()
        imdb_boxes[i] = im_proposals(net, blob)
        _t.toc()
        if (i + 1) % 10 == 0:
            print 'im_proposals: {:d}/{:d} {:.3f}s' \
                  .format(i + 1, imdb.num_images, _t.average_time)

    return imdb_boxes
