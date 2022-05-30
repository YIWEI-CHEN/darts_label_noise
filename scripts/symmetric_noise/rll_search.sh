#!/usr/bin/env bash

DATE=`date +%m%d%y`
SEED=$1
CORRUPTION_PROB=$2
GPU=$3
LOSS="rll"
EXP_PATH="logs/${DATE}_cifar10_${LOSS}_symmetric-${CORRUPTION_PROB}_gpu"

python train_search.py --data cifar10 --batchsz 64 --gpu "${GPU}" --exp_path "${EXP_PATH}" --seed "${SEED}" \
    --train_portion 0.9 --dataset cifar10 --corruption_prob "${CORRUPTION_PROB}" --corruption_type unif \
    --gold_fraction 0 --loss_func "${LOSS}" --time_limit 432000