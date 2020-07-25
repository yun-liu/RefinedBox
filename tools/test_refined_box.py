#!/usr/bin/env python

import _init_paths
import pprint
import time, sys, os
import cPickle
import scipy.io as sio
import multiprocessing as mp
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from train_refined_box import rerank_generate
import argparse
import caffe

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

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
    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    mp_queue = mp.Queue()
    rerank_generate(queue = mp_queue,
                    imdb_names = args.imdb_name,
                    rerank_model_path = args.caffemodel,
                    cfg = cfg,
                    rerank_test_prototxt = args.prototxt)
    prop_out = mp_queue.get()
    with open(prop_out['proposal_path'], 'rb') as f:
        all_boxes = cPickle.load(f)

    imdb = get_imdb(args.imdb_name)
    imdb.set_proposal_method('mat')
    imdb.config['mat_file'] = os.path.join(cfg.DATA_DIR, 'edge_boxes_data',
                                    'voc_2007_test.mat')

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    res = imdb.evaluate_recall(
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], limit = 100)
    pprint.pprint(res)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    res = imdb.evaluate_recall(candidate_boxes = all_boxes,
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], limit = 100)
    pprint.pprint(res)
