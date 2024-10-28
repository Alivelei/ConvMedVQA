# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/21 16:01
    @Author : smile 笑
    @File : mobile_net.py
    @desc : framework.model.mobile_mix.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.model.mobile_mix.network.word_embedding import WordEmbedding
from framework.model.mobile_mix.network.hidden_mix import BalancedTransMix, SoftTransMix, HardTransMix


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


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


class MobileNetV2(nn.Module):
    def __init__(self, qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="soft_hidden_mix", mix_probability=1, mix_alpha_1=3, mix_alpha_2=1):
        super().__init__()

        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, 32)

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        self.img_attn = MultiAttnFusion(32)

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.qus_linear = nn.Sequential(nn.Linear(32, 1280),
                                        nn.LayerNorm(1280),
                                        nn.GELU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.linear = nn.Linear(1280, ans_size)

    def forward(self, x, y, label=None):
        x = self.pre_conv1(x)
        y = self.qus_emb(y)

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        x = self.img_attn(x, y)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        y_out = self.qus_linear(y).mean(1)

        out = self.linear(x * y_out)

        if self.training:
            return out, mix_label
        else:
            return out

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


# 2369380 -> 2694187
def mix_hid_mobilenetv2(**kwargs):
    return MobileNetV2(qus_embedding_dim=300, ans_size=kwargs["ans_size"], glove_path=kwargs["glove_path"],
                       word_size=kwargs["word_size"], select_mix_flag=kwargs["select_mix_flag"],
                       mix_probability=kwargs["mix_probability"],  mix_alpha_1=kwargs["mix_alpha_1"],
                       mix_alpha_2=kwargs["mix_alpha_2"])


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = MobileNetV2().cuda()
    a1, a2 = model(a, b, c)
    print(a1.shape, a2.shape)
    print(sum(m.numel() for m in model.parameters()))



