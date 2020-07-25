#!/bin/bash
# Usage:
# ./experiments/scripts/refined_box.sh GPU NET DATASET [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/refined_box.sh 0 LightNet voc_2007

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
    ITERS=40000
    ;;
  voc_2012)
    TRAIN_IMDB="voc_2007_trainval+voc_2007_test+voc_2012_trainval"
    TEST_IMDB="voc_2012_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  coco)
    echo "Not implemented: use experiments/scripts/refined_box_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/refined_box_${NET}_${EXTRA_ARGS_SLUG}`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_refined_box.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/refined_box.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_refined_box.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/refined_box/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/refined_box.yml \
  ${EXTRA_ARGS}
