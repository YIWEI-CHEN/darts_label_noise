#!/usr/bin/env bash

DATE=`date +%m%d`

LOSS="cce"
ARCH="no_noise_epoch0"
GPU=3
EXP_PATH="exp/cifar10_${LOSS}_${ARCH}_gpu${GPU}"
python train.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} --epochs 75 \
    --exp_path ${EXP_PATH} --seed 1 --dataset cifar10 --drop_path_prob 0.3 \
    --gold_fraction 1.0 --loss_func ${LOSS} --arch ${ARCH}