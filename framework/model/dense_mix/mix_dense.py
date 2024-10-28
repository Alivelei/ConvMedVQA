# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/21 10:14 
    @Author : smile 笑
    @File : mix_dense.py
    @desc : framework.model.dense_mix.
"""


import torch
import torch.nn as nn
from framework.model.dense_mix.network.word_embedding import WordEmbedding
from framework.model.dense_mix.network.hidden_mix import BalancedTransMix, SoftTransMix, HardTransMix


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


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


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="hard_hidden_mix", mix_probability=1, mix_alpha_1=5, mix_alpha_2=1):
        super().__init__()
        self.growth_rate = growth_rate

        inner_channels = 2 * growth_rate

        # self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inner_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.GELU(),
        )
        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, inner_channels)

        self.img_attn = MultiAttnFusion(inner_channels)

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index),
                                     self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1),
                                 self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.qus_linear = nn.Sequential(nn.Linear(2 * growth_rate, inner_channels),
                                        nn.LayerNorm(inner_channels),
                                        nn.GELU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, ans_size)

    def forward(self, x, y, label=None):
        x = self.conv1(x)
        y = self.qus_emb(y)

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        x = self.img_attn(x, y)

        output = self.features(x)

        output = self.avgpool(output)

        output = output.view(output.size()[0], -1)

        y_out = self.qus_linear(y).mean(1)

        output = self.linear(output * y_out)

        if self.training:
            return output, mix_label
        else:
            return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


# 7048548 -> 7436235
def mix_hid_densenet121(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, qus_embedding_dim=300, ans_size=kwargs["ans_size"],
                    glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
                    select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                    mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 12643172 -> 13152459
def mix_hid_densenet169(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, qus_embedding_dim=300, ans_size=kwargs["ans_size"],
                    glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
                    select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                    mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 18277220 -> 18835147
def mix_hid_densenet201(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, qus_embedding_dim=300, ans_size=kwargs["ans_size"],
                    glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
                    select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                    mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 26681188 -> 27476683
def mix_hid_densenet161(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, qus_embedding_dim=300, ans_size=kwargs["ans_size"],
                    glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
                    select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                    mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = DenseNet(Bottleneck, [6, 12, 36, 24], 48).cuda()

    res, m_l = model(a, b, c)
    print(res.shape, m_l.shape)
    print(sum(x.numel() for x in model.parameters()))  # 36929803
