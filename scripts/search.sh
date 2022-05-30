#!/usr/bin/env bash

DATE=`date +%m%d`

#EXP_PATH="exp/${DATE}_cifar10_no_noise_gpu"
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu 0 --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 --gold_fraction 1.0 \

#EXP_PATH="exp/${DATE}_cifar10_flip_noise_07_gpu"
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu 3 --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type flip --gold_fraction 0


#EXP_PATH="exp/${DATE}_cifar10_uniform_noise_07_gpu"
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu 2 --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0

#LOSS="rll"
#EXP_PATH="exp/${DATE}_cifar10_${LOSS}_uniform_noise_07_gpu"
#GPU=0
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

#LOSS="rll"
#EXP_PATH="exp/${DATE}_cifar10_${LOSS}_flip_noise_07_gpu"
#GPU=3
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type flip --gold_fraction 0 --loss_func ${LOSS}

#LOSS="rll"
#EXP_PATH="exp/${DATE}_cifar10_${LOSS}_no_noise_gpu"
#GPU=2
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} --model_path exp \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --gold_fraction 1.0 --loss_func ${LOSS}

#LOSS="cce"
#EXP_PATH="exp/${DATE}_cifar10_${LOSS}_uniform_07_gpu"
#GPU=2
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} --time_limit 432000

#LOSS="rll"
#EXP_PATH="exp/${DATE}_cifar10_${LOSS}_uniform_07_gpu"
#GPU=0
#python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} \
#    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} --time_limit 432000

LOSS="cce"
EXP_PATH="exp/${DATE}_cifar10_${LOSS}_uniform_07_clean_valid_gpu"
GPU="0,1,2,3"
python train_search.py --data cifar10 --batchsz 64 --lr 0.1 --wd 0.0005 --gpu ${GPU} \
    --exp_path ${EXP_PATH} --seed 1 --train_portion 0.9 --dataset cifar10 \
    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} --time_limit 432000 \
    --clean_valid
