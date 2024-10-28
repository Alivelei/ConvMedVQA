# _*_ coding: utf-8 _*_

"""
    @Time : 2023/5/31 13:48 
    @Author : smile 笑
    @File : hidden_mix.py
    @desc :
"""


import torch
from timm.data.mixup import Mixup, one_hot
import numpy as np
from torch import nn


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda', return_y1y2=False):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)  # flip翻转

    if return_y1y2:
        return y1 * lam + y2 * (1. - lam), y1.clone(), y2.clone()
    else:
        return y1 * lam + y2 * (1. - lam)


def rand_qus_box(size, lam):
    L = size[1]

    cut_l = np.int(L * (1. - lam))
    cl = np.random.randint(L)
    bbz1 = np.clip(cl - cut_l // 2, 0, L)
    bbz2 = np.clip(cl + cut_l // 2, 0, L)

    return bbz1, bbz2


class BalancedTransMix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0, label_smoothing=0.1, num_classes=1000):
        super(BalancedTransMix, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def self_params_per_batch(self):
        lam = 1.
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_beta)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam

    def multi_mix_batch(self, x, y):
        lam = self.self_params_per_batch()

        x_bbz1, x_bbz2 = rand_qus_box(x.size(), lam)

        x_flipped = x[:, x_bbz1:x_bbz2].flip(0).mul_(1. - lam)
        x[:, x_bbz1:x_bbz2].mul_(lam).add_(x_flipped)

        y_bbz1, y_bbz2 = rand_qus_box(y.size(), lam)

        y_flipped = y[:, y_bbz1:y_bbz2].flip(0).mul_(1. - lam)
        y[:, y_bbz1:y_bbz2].mul_(lam).add_(y_flipped)

        return lam

    def __call__(self, x, y, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self.multi_mix_batch(x, y)  # tuple or value

        mixed_target, y1, y2 = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device, return_y1y2=True)  # tuple or tensor

        return mixed_target, lam


class SoftTransMix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0, label_smoothing=0.1, num_classes=1000):
        super(SoftTransMix, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def self_params_per_batch(self):
        lam = 1.
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_beta)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam

    def multi_mix_batch(self, x, y):
        lam = self.self_params_per_batch()

        x_flipped = x.flip(0).mul_(1. - lam)
        y_flipped = y.flip(0).mul_(1. - lam)

        x.mul_(lam).add_(x_flipped)
        y.mul_(lam).add_(y_flipped)

        return lam

    def __call__(self, x, y, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self.multi_mix_batch(x, y)  # tuple or value

        mixed_target, y1, y2 = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device,
                                            return_y1y2=True)  # tuple or tensor

        return mixed_target, lam


class HardTransMix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0, label_smoothing=0.1, num_classes=1000):
        super(HardTransMix, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def self_params_per_batch(self):
        lam = 1.
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_beta)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam

    def multi_mix_batch(self, x, y):
        lam = self.self_params_per_batch()

        x_bbz1, x_bbz2 = rand_qus_box(x.size(), lam)
        print(x_bbz1, x_bbz2)
        x_flipped = x[:, x_bbz1:x_bbz2].flip(0)
        x[:, x_bbz1:x_bbz2] = x_flipped

        y_bbz1, y_bbz2 = rand_qus_box(y.size(), lam)

        y_flipped = y[:, y_bbz1:y_bbz2].flip(0)
        y[:, y_bbz1:y_bbz2] = y_flipped

        return lam

    def __call__(self, x, y, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self.multi_mix_batch(x, y)  # tuple or value

        mixed_target, y1, y2 = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device,
                                            return_y1y2=True)  # tuple or tensor

        return mixed_target, lam


