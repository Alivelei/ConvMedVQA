# _*_ coding: utf-8 _*_

"""
    @Time : 2021/11/4 18:48 
    @Author : smile 笑
    @File : mixup.py
    @desc :
"""


import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_qus_box(size, lam):
    L = size[1]

    cut_l = np.int(L * (1. - lam))
    cl = np.random.randint(L)
    bbz1 = np.clip(cl - cut_l // 2, 0, L)
    bbz2 = np.clip(cl + cut_l // 2, 0, L)

    return bbz1, bbz2


def cut_img_qus_mixup(model, image, qus, ans, criterion, mix_alpha1=5, mix_alpha2=1):
    lam = np.random.beta(mix_alpha1, mix_alpha2)  # 通过lam决定裁剪叠加块的大小，并在后面计算loss时作为权重
    rand_index = torch.randperm(image.size()[0]).cuda()
    target_a = ans
    target_b = ans[rand_index]

    # image cut mix
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]  # 进行裁剪替换操作

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

    # qus cut mix
    bbz1, bbz2 = rand_qus_box(qus.size(), lam)
    qus[:, bbz1: bbz2] = qus[rand_index, bbz1: bbz2]

    # compute output
    output = model(image, qus)
    loss = criterion(output, target_a.view(-1)) * lam + criterion(output, target_b.view(-1)) * (1. - lam)  # 以lam作为权重

    return loss, output


def img_qus_mixup(model, image, qus, ans, criterion, mix_alpha1=5, mix_alpha2=1):
    # 1.mixup 使用mixup数据增强方式进行数据增强
    lam = np.random.beta(mix_alpha1, mix_alpha2)

    # randperm返回1~images.size(0)的一个随机排列
    index = torch.randperm(image.size(0)).cuda()
    inputs_a = lam * image + (1 - lam) * image[index, :]

    mask_b = torch.rand_like(qus, dtype=torch.float32) < (1 - lam)  # 使用概率来决定每个位置是否需要替换单词
    inputs_b = torch.masked_scatter(qus, mask_b, qus[index])

    ans_a, ans_b = ans, ans[index]
    predict_ans = model(inputs_a, inputs_b)

    ans_loss = lam * criterion(predict_ans, ans_a.view(-1)) + (1 - lam) * criterion(predict_ans, ans_b.view(-1))

    return ans_loss, predict_ans


def img_mixup(model, image, qus, ans, criterion, mix_alpha1=5, mix_alpha2=1):
    # 1.mixup 使用mixup数据增强方式进行数据增强
    lam = np.random.beta(mix_alpha1, mix_alpha2)

    # randperm返回1~images.size(0)的一个随机排列
    index = torch.randperm(image.size(0)).cuda()
    inputs_a = lam * image + (1 - lam) * image[index, :]

    ans_a, ans_b = ans, ans[index]
    predict_ans = model(inputs_a, qus)

    ans_loss = lam * criterion(predict_ans, ans_a.view(-1)) + (1 - lam) * criterion(predict_ans, ans_b.view(-1))

    return ans_loss, predict_ans


def cut_img_mixup(model, image, qus, ans, criterion, mix_alpha1=5, mix_alpha2=1):
    lam = np.random.beta(mix_alpha1, mix_alpha2)  # 通过lam决定裁剪叠加块的大小，并在后面计算loss时作为权重
    rand_index = torch.randperm(image.size()[0]).cuda()
    target_a = ans
    target_b = ans[rand_index]

    # image cut mix
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]  # 进行裁剪替换操作

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

    # compute output
    output = model(image, qus)
    loss = criterion(output, target_a.view(-1)) * lam + criterion(output, target_b.view(-1)) * (1. - lam)  # 以lam作为权重

    return loss, output


if __name__ == '__main__':
    from torch import nn
    a = torch.randn([2, 3, 224, 224])
    b = torch.randint(0, 100, [2, 20], dtype=torch.int64)
    ans = torch.ones([2], dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()

    cut_img_mixup(1, a, b, ans, criterion)

