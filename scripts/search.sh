#!/usr/bin/env bash

DATE=`date +%m%d`
python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu 0 --model_path exp --exp_path "exp/cifar10-${DATE}-" --seed 1 --train_portion 0.9
