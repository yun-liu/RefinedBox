"""
Demo script showing detections in a database.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from datasets.factory import get_imdb
from utils.timer import Timer
import numpy as np
import caffe, os, cv2
import argparse
import pprint

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      (255, 0, 0),
                      3)
        cv2.putText(im,
                    '{:s} {:.2f}'.format(class_name, score),
                    (int(bbox[0] + 5), int(bbox[1] + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

    return im

def visualize_net(net, imdb):
    """Test and visualize a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    output_dir = get_output_dir(imdb, net)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        #inds = np.where(roidb[i]['gt_classes'] != 0)[0]
        #classes = np.unique(roidb[i]['gt_classes'][inds])

        _t['misc'].tic()
        # Visualize detections for each class
        for cls in CLASSES[1:]:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            im = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        _t['misc'].toc()

        cv2.imwrite(os.path.join(output_dir, os.path.basename(\
                        imdb.image_path_at(i))[:-4] + '_rpn.jpg'), im)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not os.path.isfile(args.caffemodel):
        raise IOError(('{:s} not found.').format(args.caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    if cfg.TEST.REFINED_BOX:
        imdb.set_proposal_method('mat')
        imdb.config['mat_file'] = os.path.join(cfg.DATA_DIR, cfg.TEST.RERANK_PROP_PATH,
                                        args.imdb_name + '.mat')

    visualize_net(net, imdb)
