#!/usr/bin/env python

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import numpy as np
import sys, os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a RefinedBox network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, mat_file=None, rerank_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if mat_file is not None:
        imdb.config['mat_file'] = mat_file
    if rerank_file is not None:
        imdb.config['pkl_file'] = rerank_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

def _init_caffe(cfg):
    """
    Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def train_rerank(imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """
    Train a Region Proposal Network in a separate training process.
    """

    cfg.TRAIN.PROPOSAL_METHOD = 'mat'
    cfg.TRAIN.FG_THRESH = 0.7
    cfg.TRAIN.BG_THRESH_HI = 0.3
    cfg.TRAIN.BG_THRESH_LO = 0.0
    cfg.TRAIN.SNAPSHOT_ITERS = 10000
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    mat_file = os.path.join(cfg.DATA_DIR, 'edge_boxes_data', imdb_name+'.mat')
    roidb, imdb = get_roidb(imdb_name, mat_file=mat_file)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    print 'Final model: {}'.format(model_paths[-1])

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    cfg.EXP_DIR = 'deep_box'
    cfg.TRAIN.REFINED_BOX = True
    cfg.TRAIN.RERANK_PROPOSAL = True

    # solver and test_prototxt for DeepBox
    solver = os.path.join(cfg.MODELS_DIR, 'Alex', 'deep_box', 'solver.prototxt')
    test_prototxt = os.path.join(cfg.MODELS_DIR, 'Alex', 'deep_box', 'test.prototxt')

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'RERANK, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    train_rerank(imdb_name=args.imdb_name, init_model=args.pretrained_model,
        solver=solver, max_iters=60000, cfg=cfg)
