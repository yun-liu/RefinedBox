#!/usr/bin/env python

import _init_paths
import pprint
import sys, os
import cPickle
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
from fast_rcnn.test import test_net
from train_refined_box_alt_opt import rerank_generate
import caffe


# pre-defined variables
test_imdb = 'voc_2007_test'
image_set_file = 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
out_path = '/media/liuyun/E60A05E00A05AF1D/LiuYun/Project/Objectness/Data/boxes_voc2007/RefinedBoxSS/'


# Use the pre-trained model to re-rank proposals
cfg.GPU_ID = 0
cfg.EXP_DIR = 'boxes'

cfg.TEST.RERANK_PROPOSAL = True
cfg.TEST.REFINED_BOX = True
cfg.TEST.RERANK_MIN_SIZE = 1
cfg.CACHE = 'cache'
cfg.TRAIN.RERANK_PROP_PATH = 'refined_box/mat_data'
mp_queue = mp.Queue()
model_path = 'output/refined_box_alt_opt_VOC07_SectiveSearch_70.7/voc_2007_trainval/VGG16_refined_box_final.caffemodel'
test_prototxt = 'models/pascal_voc/VGG16/refined_box_alt_opt/rerank_test.prototxt'
rerank_generate(queue=mp_queue,
                imdb_names=test_imdb,
                rerank_model_path=model_path,
                cfg=cfg,
                rerank_test_prototxt=test_prototxt)

# load re-ranked proposals and ecaluate them
det_file = 'output/{}/{}/VGG16_refined_box_final_{}_proposals.pkl'.format(cfg.EXP_DIR, test_imdb, test_imdb)
with open(det_file, 'rb') as f:
    all_boxes = cPickle.load(f)

imdb = get_imdb(test_imdb)
imdb.set_proposal_method('mat')
imdb.config['mat_file'] = os.path.join(cfg.DATA_DIR, 'refined_box/mat_data',
                                '{}.mat'.format(test_imdb))

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
res = imdb.evaluate_recall(thresholds=[0.5, 0.7], limit=100)
pprint.pprint(res)

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
res = imdb.evaluate_recall(candidate_boxes=all_boxes, thresholds=[0.5, 0.7], limit=100)
pprint.pprint(res)

# save the proposals into 'mat' format
sio.savemat(det_file[:-4] + '.mat', {'boxes': all_boxes})
if not os.path.exists(out_path):
    os.makedirs(out_path)
with open(image_set_file, 'r') as f:
    image_index = [x.strip() for x in f.readlines()]
for idx in range(len(image_index)):
    sio.savemat(os.path.join(out_path, image_index[idx] + '.mat'), {'proposals': {'boxes': all_boxes[idx] + 1}})
