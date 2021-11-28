# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

#DIR="/home/nikhar/330Project/data/fungi/images/"
DIR="~/330/330Project/data/fungi/train/"
LOGDIR="~/330/330Project/tmp"
ARCH="alexnet"
LR=0.05
WD=-5
K=762
WORKERS=2

CUDA_VISIBLE_DEVICES=0 python main.py ${DIR} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --log_dir LOGDIR
