# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/18 10:55 
    @Author : smile 笑
    @File : mix_resnet_hid.py
    @desc : framework.model.resnet_mix.
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
from framework.model.resnet_mix.network.word_embedding import WordEmbedding
from framework.model.resnet_mix.network.hidden_mix2 import BalancedTransMix, SoftTransMix, HardTransMix


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QusEmbeddingMap(nn.Module):
    def __init__(self, glove_path, word_size, embedding_dim, hidden_size):
        super(QusEmbeddingMap, self).__init__()

        self.embedding = WordEmbedding(word_size, embedding_dim, 0.0, False)
        self.embedding.init_embedding(glove_path)

        self.linear = nn.Linear(embedding_dim, hidden_size)

    def forward(self, qus):
        text_embedding = self.embedding(qus)

        text_x = self.linear(text_embedding)

        return text_x


class MultiAttnFusion(nn.Module):
    def __init__(self, emb_dim=64, dropout=.0):
        super(MultiAttnFusion, self).__init__()

        self.scale = emb_dim ** -0.5
        self.attn_drop = nn.Dropout(dropout)

        self.layer_text = nn.LayerNorm(emb_dim)
        self.tanh_gate = nn.Linear(emb_dim, emb_dim)
        self.sigmoid_gate = nn.Linear(emb_dim, emb_dim)

        self.conv_end = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x, y):
        b, c, h, w = x.shape

        query = x.flatten(2).transpose(1, 2).contiguous()  # b s d   b q d    bsd bdq    bsq bqd   bsd
        key = self.layer_text(y)
        concated = torch.mul(torch.tanh(self.tanh_gate(y)), torch.sigmoid(self.sigmoid_gate(y)))

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        end_x = (attn @ concated).transpose(1, 2).contiguous().reshape(b, c, h, w)
        end_x = self.conv_end(end_x) + x

        return end_x


class ResNetMedVQAModel(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=[3, 4, 23, 3], qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="bal_hidden_mix", mix_probability=1, mix_alpha_1=5, mix_alpha_2=1):
        super(ResNetMedVQAModel, self).__init__()
        self.in_planes = 64

        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, 64)
        self.img_attn = MultiAttnFusion(64)

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.qus_linear = nn.Sequential(nn.Linear(64, 512 * block.expansion),
                                        nn.LayerNorm(512 * block.expansion),
                                        nn.GELU())
        self.linear = nn.Linear(512*block.expansion, ans_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y, label=None):
        y = self.qus_emb(y)

        x = F.gelu(self.bn1(self.conv1(x)))

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        out = self.img_attn(x, y)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7, 7)
        out = out.view(out.size(0), -1)

        y_out = self.qus_linear(y).mean(1)

        out = self.linear(out * y_out)

        if self.training:
            return out, mix_label
        else:
            return out


def mix_hid_resnet34_base(**kwargs):
    return ResNetMedVQAModel(
        block=BasicBlock, num_blocks=[3, 4, 6, 3], qus_embedding_dim=300, ans_size=kwargs["ans_size"],
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], select_mix_flag=kwargs["select_mix_flag"],
        mix_probability=kwargs["mix_probability"], mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"]
    )


def mix_hid_resnet50_base(**kwargs):
    model = ResNetMedVQAModel(
        block=Bottleneck, num_blocks=[3, 4, 6, 3], qus_embedding_dim=300,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
        ans_size=kwargs["ans_size"], select_mix_flag=kwargs["select_mix_flag"],
        mix_probability=kwargs["mix_probability"], mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"]
    )
    return model


def mix_hid_resnet101_base(**kwargs):  # 43279627
    model = ResNetMedVQAModel(
        block=Bottleneck, num_blocks=[3, 4, 23, 3], qus_embedding_dim=300,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
        ans_size=kwargs["ans_size"], select_mix_flag=kwargs["select_mix_flag"],
        mix_probability=kwargs["mix_probability"], mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"]
    )
    return model


def mix_hid_resnet152_base(**kwargs):
    model = ResNetMedVQAModel(
        block=Bottleneck, num_blocks=[3, 8, 36, 3], qus_embedding_dim=300,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
        ans_size=kwargs["ans_size"], select_mix_flag=kwargs["select_mix_flag"],
        mix_probability=kwargs["mix_probability"], mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"]
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = ResNetMedVQAModel().cuda()

    res, m_l = model(a, b, c)
    print(res.shape, m_l.shape)
    print(sum(x.numel() for x in model.parameters()))  # 43279627







