# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/21 17:54 
    @Author : smile 笑
    @File : vgg_net.py
    @desc : framework.model.vgg_mix.
"""


import torch
import torch.nn as nn
from framework.model.vgg_mix.network.word_embedding import WordEmbedding
from framework.model.vgg_mix.network.hidden_mix import BalancedTransMix, SoftTransMix, HardTransMix
import torch.nn.functional as F


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


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


class VGG(nn.Module):
    def __init__(self, features, qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="soft_hidden_mix", mix_probability=1, mix_alpha_1=3, mix_alpha_2=1):
        super().__init__()
        self.features = features

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, 64)
        self.img_attn = MultiAttnFusion(64)

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        self.qus_linear = nn.Sequential(nn.Linear(64, 512),
                                        nn.LayerNorm(512),
                                        nn.GELU())

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, ans_size)
        )

    def forward(self, x, y, label=None):
        x = self.conv1(x)
        y = self.qus_emb(y)

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        x = self.img_attn(x, y)

        output = self.features(x)

        output = F.adaptive_avg_pool2d(output, (1, 1))

        output = output.view(output.size()[0], -1)

        y_out = self.qus_linear(y).mean(1)
        output = self.classifier(output * y_out)

        if self.training:
            return output, mix_label
        else:
            return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 64
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


# 28518244 -> 29286475
def mix_hid_vgg11_bn(**kwargs):
    return VGG(make_layers(cfg['A'], batch_norm=True), qus_embedding_dim=300, ans_size=kwargs["ans_size"],
               glove_path=kwargs["glove_path"], word_size=kwargs["word_size"])


# 28703140 -> 29471371
def mix_hid_vgg13_bn(**kwargs):
    return VGG(make_layers(cfg['B'], batch_norm=True), qus_embedding_dim=300, ans_size=kwargs["ans_size"],
               glove_path=kwargs["glove_path"], word_size=kwargs["word_size"])


# 34015396 -> 34783627
def mix_hid_vgg16_bn(**kwargs):
    return VGG(make_layers(cfg['D'], batch_norm=True), qus_embedding_dim=300, ans_size=kwargs["ans_size"],
               glove_path=kwargs["glove_path"], word_size=kwargs["word_size"])


# 39327652 -> 40095883
def mix_hid_vgg19_bn(**kwargs):
    return VGG(make_layers(cfg['E'], batch_norm=True), qus_embedding_dim=300, ans_size=kwargs["ans_size"],
               glove_path=kwargs["glove_path"], word_size=kwargs["word_size"])


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = VGG(make_layers(cfg['E'], batch_norm=True)).cuda()

    res, m_l = model(a, b, c)
    print(res.shape, m_l.shape)
    print(sum(x.numel() for x in model.parameters()))







