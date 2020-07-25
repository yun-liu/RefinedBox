#!/usr/bin/env python

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from refined_box.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

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

def combined_roidb(imdb_names, mat_folder=None, pkl_folder=None):
    def get_roidb(imdb_name, mat_folder, pkl_folder):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        if mat_folder is not None:
            mat_file = os.path.join(cfg.DATA_DIR, mat_folder, imdb_name+'.mat')
            imdb.config['mat_file'] = mat_file
        if pkl_folder is not None:
            pkl_file = pkl_folder + imdb_name + '_proposals.pkl'
            imdb.config['pkl_file'] = pkl_file
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s, mat_folder, pkl_folder) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

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

def train_rerank(queue=None, imdb_names=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """
    Train a Region Proposal Network in a separate training process.
    """

    cfg.TRAIN.RERANK_PROPOSAL = True
    cfg.TRAIN.PROPOSAL_METHOD = 'mat'
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.TRAIN.FG_FRACTION = 0.25
    cfg.TRAIN.FG_THRESH = 0.7
    cfg.TRAIN.BBOX_THRESH = 0.7
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    imdb, roidb = combined_roidb(imdb_names, mat_folder='edge_boxes_data')
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model = init_model,
                            max_iters = max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rerank_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rerank_model_path})

def rerank_generate(queue=None, imdb_names=None, rerank_model_path=None, cfg=None,
                 rerank_test_prototxt=None):
    """Use a trained RERANK to generate proposals."""

    cfg.TEST.RERANK_PRE_NMS_TOP_N = -1     # no pre NMS filtering
    cfg.TEST.RERANK_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
    print 'RERANK model: {}'.format(rerank_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # Load RERANK and configure output directory
    if len(imdb_names.split('+')) > 1:
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    rerank_net = caffe.Net(rerank_test_prototxt, rerank_model_path, caffe.TEST)

    for imdb_name in imdb_names.split('+'):
        imdb = get_imdb(imdb_name)
        imdb.set_proposal_method('mat')
        imdb.config['mat_file'] = os.path.join(cfg.DATA_DIR, 'edge_boxes_data',
                                        imdb_name+'.mat')
        print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

        # Generate proposals on the imdb
        rerank_proposals = imdb_proposals(rerank_net, imdb)
        # Write proposals to disk and send the proposal file path through the
        # multiprocessing queue
        rerank_net_name = os.path.splitext(os.path.basename(rerank_model_path))[0]
        rerank_proposals_path = os.path.join(
            output_dir, rerank_net_name + '_' + imdb_name + '_proposals.pkl')
        with open(rerank_proposals_path, 'wb') as f:
            cPickle.dump(rerank_proposals, f, cPickle.HIGHEST_PROTOCOL)
        print 'Wrote RERANK proposals to {}'.format(rerank_proposals_path)
    queue.put({'proposal_path': rerank_proposals_path})

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    # queue for communicated results between processes
    mp_queue = mp.Queue()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'RERANK, init from scratch'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    solver = os.path.join(cfg.MODELS_DIR, args.net_name, 'refined_box',
            'solver.prototxt')
    train_rerank(queue = mp_queue, imdb_names = args.imdb_name,
            init_model = args.pretrained_model, solver = solver,
            max_iters = 80000, cfg = cfg)
    rerank_out = mp_queue.get()

    # Create final model (just a copy of the last stage)
    final_path = os.path.join(
            os.path.dirname(rerank_out['model_path']),
            args.net_name + '_refined_box_final.caffemodel')
    print 'cp {} -> {}'.format(rerank_out['model_path'], final_path)
    shutil.copy(rerank_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)
