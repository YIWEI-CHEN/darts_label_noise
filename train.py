import argparse
import glob
import logging
import sys
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

import utils
from model import NetworkCIFAR as Network
from load_corrupted_data import CIFAR10, CIFAR100
import genotypes

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=96, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='logs/cifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--arch', type=str, default='resnet18', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=1.0, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif", "flip", hierarchical).')
parser.add_argument('--loss_func', type=str, default='rll', choices=['cce', 'rll'],
                    help='Choose between Categorical Cross Entropy (CCE), Robust Log Loss (RLL).')
args = parser.parse_args()

args.save = args.exp_path + '-' + time.strftime("%m%d")
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')


def main():
    torch.cuda.set_device(device)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # Load dataset
    if args.dataset == 'cifar10':
        if args.gold_fraction == 0:
            train_data = CIFAR10(
                root=args.data, train=True, gold=False, gold_fraction=args.gold_fraction,
                corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
                transform=train_transform, download=True, seed=args.seed)
        else:
            train_data = CIFAR10(
                root=args.data, train=True, gold=True, gold_fraction=args.gold_fraction,
                corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
                transform=train_transform, download=True, seed=args.seed)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        num_classes = 10

    elif args.dataset == 'cifar100':
        if args.gold_fraction == 0:
            train_data = CIFAR100(
                root=args.data, train=True, gold=False, gold_fraction=args.gold_fraction,
                corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
                transform=train_transform, download=True, seed=args.seed)
        else:
            train_data = CIFAR100(
                root=args.data, train=True, gold=True, gold_fraction=args.gold_fraction,
                corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
                transform=train_transform, download=True, seed=args.seed)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        num_classes = 100

    if "resnet" in args.arch:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).cuda()
    else:
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(args.init_ch, num_classes, args.layers, args.auxiliary, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.loss_func == 'cce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss_func == 'rll':
        criterion = utils.RobustLogLoss().to(device)
    else:
        assert False, "Invalid loss function '{}' given. Must be in {'cce', 'rll'}".format(args.loss_func)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True
    )

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc: %f', valid_acc)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc: %f', train_acc)

        utils.save(model, os.path.join(args.save, 'trained.pt'))
        print('saved to: trained.pt')


def train(train_queue, model, criterion, optimizer):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        if "resnet" in args.arch:
            logits, logits_aux = model(x), 0
            loss = criterion(logits, target)
        else:
            logits, logits_aux = model(x)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            if "resnet" in args.arch:
                logits = model(x)
            else:
                logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
