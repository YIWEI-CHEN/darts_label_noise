#!/usr/bin/env bash

DATE=`date +%m%d%y_%H%M%S`
SEED=$1
CORRUPTION_PROB=$2
GPU=$3
ARCH=$4
LOSS="rll"

if [[ "${ARCH}" =~ "resnet" ]]
then
  EXP_PATH="logs/train/${DATE}_seed-${SEED}_cifar10_${ARCH}_${LOSS}_symmetric-${CORRUPTION_PROB}_gpu"
  python train.py --data cifar10 --batchsz 96 --lr 0.025 --wd 3e-4 --gpu "${GPU}" --epochs 600 \
  --exp_path "${EXP_PATH}" --seed "${SEED}" --corruption_prob "${CORRUPTION_PROB}" --corruption_type unif \
  --gold_fraction 0 --loss_func "${LOSS}" --cutout --arch "${ARCH}"
else
  EXP_PATH="logs/train/${DATE}_seed-${SEED}_cifar10_${ARCH}_gpu"
  python train.py --data cifar10 --batchsz 96 --lr 0.025 --wd 3e-4 --gpu "${GPU}" --epochs 600 \
  --exp_path "${EXP_PATH}" --seed "${SEED}" --corruption_prob "${CORRUPTION_PROB}" --corruption_type unif \
  --gold_fraction 0 --loss_func "${LOSS}" --auxiliary --cutout --arch "${ARCH}"
fi
