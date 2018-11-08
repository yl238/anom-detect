#!/bin/bash

# set this to your python binary
export PYTHONBIN='python'
export PYTHONUNBUFFERED=1

# set exp root dir and dataset root dir
export DATA_DIR='../data/tar_dir/'
export EXP_DIR='../exp_results'

# train
$PYTHONBIN ../train.py --data ${DATA_DIR} --cuda \
    --epochs 41 --lr 1e-4 --batch_size 32 --out_dir ${EXP_DIR}/NORMAL0.1 \
    --image_size 128 --kl_weight 0.01
