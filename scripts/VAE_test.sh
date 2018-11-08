#!/usr/bin/env bash

# set this to your python binary
export PYTHONBIN='python'
export PYTHONUNBUFFERED=1

# set exp root dir and dataset root dir
export DATA_DIR='../data/tar_dir/'
export EXP_DIR='../exp_results/'

# test
$PYTHONBIN ../outlier_detection.py --cuda \
    --data ${DATA_DIR} \
    --model_path ${EXP_DIR}/NV_kl0.01/best_model.pth.tar \
    --image_size 128 --kl_weight 0.01 --out_csv auc_result5.csv
