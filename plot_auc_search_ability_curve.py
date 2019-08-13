import os

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from sklearn import metrics

parser = argparse.ArgumentParser("auc_sac")
parser.add_argument('--log', type=str, help='location of the log file')
parser.add_argument('--sac_name', type=str, help='name of search ability curve')
parser.add_argument('--title', type=str, help='title of the figure')
parser.add_argument('--clean', type=str, default='', help='log file of clean')
parser.add_argument('--cce_unif', type=str, default='', help='log file of CCE + uniform noise')
parser.add_argument('--rll_unif', type=str, default='', help='log file of RLL + uniform noise')
args = parser.parse_args()

def get_validation_accuracy(file):
    pattern = re.compile(r'(?<=valid acc: )[0-9]{1,2}\.[0-9]{1,6}')
    accuracy = list()
    with open(file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match is None:
                accuracy.append(float(match.group())/100)
    # print('\n'.join(map(str, accuracy)))
    return np.asarray(accuracy)


def auc_search_ability_curve(x, y):
    auc = metrics.auc(x, y)
    init_acc = np.full(y.shape, y[0])
    offset = metrics.auc(x, init_acc)
    return auc - offset


def plot_search_ability_curve(x, y):
    plt.figure()
    auc_sac = auc_search_ability_curve(x, y)
    dir_name = os.path.dirname(args.log)
    fig_name = os.path.join(dir_name, 'search_ability_curve_{}'.format(args.sac_name))
    plt.plot(x, y, label='SA curve - {} (area = {:.2f})'.format(args.sac_name.replace('_', ' '), auc_sac))
    init_acc = np.full(y.shape, y[0])
    plt.plot(x, init_acc, label='Init Arch.')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(args.title)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


def plot_search_ability_curves():
    plt.figure()
    dir_name = '/home/yiwei'
    fig_name = os.path.join(dir_name, 'search_ability_curve_{}'.format('all'))
    title = 'Search Ability Curve (DARTS)'

    for tag, log in [
        ('clean', args.clean),
        ('uniform_CCE', args.cce_unif),
        ('uniform_RLL', args.rll_unif),
    ]:
        y = get_validation_accuracy(log)
        x = np.arange(len(y))
        auc_sac = auc_search_ability_curve(x, y)
        plt.plot(x, y, label='SA curve - {} (area = {:.2f})'.format(tag, auc_sac))
        # init_acc = np.full(y.shape, y[0])
        # plt.plot(x, init_acc, label='Init Arch. - {}'.format(tag))

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

if __name__ == '__main__':
    # valid_acc = get_validation_accuracy(args.log)
    # x = np.arange(len(valid_acc))
    # auc_sac = auc_search_ability_curve(x, valid_acc)
    # print(auc_sac)
    # plot_search_ability_curve(x, valid_acc)
    plot_search_ability_curves()