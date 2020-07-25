#!/bin/bash
# Usage:
# ./experiments/scripts/refined_box_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/refined_box_alt_opt.sh 0 VGG16 voc_2007

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  voc_2007)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ;;
  voc_2012)
    TRAIN_IMDB="voc_2007_trainval+voc_2007_test+voc_2012_trainval"
    TEST_IMDB="voc_2012_test"
    PT_DIR="pascal_voc"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train_in_voc"
    TEST_IMDB="coco_2014_val_in_voc"
    PT_DIR="coco"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/refined_box_alt_opt_${NET}_${EXTRA_ARGS_SLUG}`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_refined_box_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/refined_box_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/refined_box_alt_opt/refined_box_test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/refined_box_alt_opt.yml \
  ${EXTRA_ARGS}
