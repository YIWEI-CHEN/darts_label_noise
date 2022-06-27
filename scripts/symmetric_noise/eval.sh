#!/usr/bin/env bash

PROJECT_ROOT="darts_label_noise/logs/train"
GPU=$1
EXP_PATH=$2

python test.py --data cifar10 --batchsz 96 --gpu "${GPU}" --exp_path "${PROJECT_ROOT}/${EXP_PATH}"
