#!/usr/bin/env python

"""Count the number of objects across different sizes"""

import _init_paths
import sys
import cv2
import argparse
import numpy as np
from datasets.factory import get_imdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Count the number of objects')
    parser.add_argument('--base', dest='base_size', help='base size to use',
                        default=7, type=int)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to count',
                        default='voc_2007_test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    obj_num = 0
    obj_num_size = np.zeros(36, dtype = int)
    count = 0

    imdb = get_imdb(args.imdb_name)
    imdb.set_proposal_method('gt')
    for i in xrange(imdb.num_images):
        boxes = imdb.roidb[i]['boxes'].astype(np.float)
        im = cv2.imread(imdb.image_path_at(i))
        height, width = im.shape[:2]
        imax = max(height, width)
        imin = min(height, width)
        ratio = 600 / imin
        if imax * ratio > 1000:
            ratio = 1000 / imax
        boxes *= ratio
        obj_num += boxes.shape[0]
        iw = boxes[:, 2] - boxes[:, 0] + 1
        ih = boxes[:, 3] - boxes[:, 1] + 1
        area = iw*ih
        count += np.sum(area > 36*49*256)
        iw = np.minimum(np.maximum(np.log(iw / args.base_size) / np.log(2.0), 1), 6)
        ih = np.minimum(np.maximum(np.log(ih / args.base_size) / np.log(2.0), 1), 6)
        index = (np.round(ih) - 1) * 6 + (np.round(iw) - 1)
        for ind in index:
            obj_num_size[ind] += 1

    print 'obj_num = {:d}'.format(obj_num)
    print 'obj_num_size = '
    print obj_num_size
    print 'count = {:d}'.format(count)
