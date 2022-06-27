# On Robustness of Neural Architecture Search Under Label Noise
This repository is the official implementation of [On Robustness of Neural Architecture Search Under Label Noise](https://doi.org/10.3389/fdata.2020.00002).



## Requirements
We use python 3.8.10 and NVIDIA GTX 1080 Ti.
```setup
pip install -r requiresments.txt
```


## Dataset
We verify our method on CIFAR-10 and CIFAR-100.
These datasets are downloaded automatically if they are unavailable under "cifar10/cifar100" directory. 


## Search
We searhc neural architectures under two types of label noise: (1) symmetric noise (2) hierarchical noise.
The noise levels are 0.2, 0.4, 0.6.
The loss function could be robust log loss (RLL) or categorical cross entropy (CCE).
The seeds are "1", "2019", "1989".
The log files would be saved under "logs/DATE_SEED_DATASET_LOSS_NOISE_GPU".

```search
# RLL (script <seed> <noise_level> <gpu>) 
sh scripts/symmetric_noise/rll_search.sh 1 0.6 0

# CCE (script <seed> <noise_level> <gpu>)
sh scripts/symmetric_noise/cce_search.sh 1989 1.0 3
```


## Training
```train
bash scripts/symmetric_noise/retrain.sh 1 1.0 0 resnet18

bash scripts/symmetric_noise/retrain.sh 2019 1.0 0 darts_rll_seed_2019_symmetric_1
bash scripts/symmetric_noise/retrain.sh 1 1.0 1 darts_rll_seed_1_symmetric_1
bash scripts/symmetric_noise/retrain.sh 1989 1.0 2 darts_rll_seed_1989_symmetric_1

bash scripts/symmetric_noise/retrain.sh 1 1.0 3 darts_cce_seed_1_symmetric_1
bash scripts/symmetric_noise/retrain.sh 1989 1.0 0 darts_cce_seed_1989_symmetric_1
bash scripts/symmetric_noise/retrain.sh 2019 1.0 1 darts_cce_seed_2019_symmetric_1
```

## Evaluation
```test
bash scripts/eval.sh 0 060922_104058_seed-2019_cifar10_darts_rll_seed_2019_symmetric_1_gpu-0609
bash scripts/eval.sh 1 060922_104203_seed-1_cifar10_darts_rll_seed_1_symmetric_1_gpu-0609
bash scripts/eval.sh 2 060922_104221_seed-1989_cifar10_darts_rll_seed_1989_symmetric_1_gpu-0609

bash scripts/eval.sh 0 060622_160654_seed-1989_cifar10_resnet18_rll_symmetric-1.0_gpu-0606
bash scripts/eval.sh 1 060622_160801_seed-2019_cifar10_resnet18_rll_symmetric-1.0_gpu-0606
bash scripts/eval.sh 2 061222_235819_seed-1_cifar10_resnet18_rll_symmetric-1.0_gpu-0612

bash scripts/eval.sh 0 060922_104426_seed-1_cifar10_darts_cce_seed_1_symmetric_1_gpu-0609
bash scripts/eval.sh 1 061222_235536_seed-1989_cifar10_darts_cce_seed_1989_symmetric_1_gpu-0612
bash scripts/eval.sh 2 061222_235640_seed-2019_cifar10_darts_cce_seed_2019_symmetric_1_gpu-0612
```

## Pre-trained Models

## Results


## Citation
If you use any part of this code in your research, please cite our [paper](https://doi.org/10.3389/fdata.2020.00002):
```
@article{chen2020robustness,
  title={On robustness of neural architecture search under label noise},
  author={Chen, Yi-Wei and Song, Qingquan and Liu, Xi and Sastry, PS and Hu, Xia},
  journal={Frontiers in big Data},
  volume={3},
  pages={2},
  year={2020},
  publisher={Frontiers}
}
```
