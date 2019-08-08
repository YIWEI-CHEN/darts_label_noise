import numpy as np
import os
import shutil
import torch
import torchvision.transforms as transforms


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """

    :param output: logits, [b, classes]
    :param target: [b]
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """

    :param args:
    :return:
    """
    CIFAR_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]  # [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [x / 255 for x in [63.0, 62.1, 66.7]]  # [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    """
    count all parameters excluding auxiliary
    :param model:
    :return:
    """
    return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    print('load from model:', model_path)
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class RobustLogLoss(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super(RobustLogLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        target_one_hot = torch.zeros(input.size())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target.unsqueeze(-1), 1)
        target_one_hot = target_one_hot.float()

        _input = torch.log(self.alpha + torch.nn.functional.softmax(input))
        loss = torch.log((self.alpha + 1) / self.alpha) - \
               target_one_hot * _input + (1 - target_one_hot) * _input / (target_one_hot.shape[1] - 1)
        return torch.mean(torch.sum(loss, dim=1))