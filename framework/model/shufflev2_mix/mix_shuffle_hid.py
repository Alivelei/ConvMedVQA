# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/18 10:22 
    @Author : smile 笑
    @File : mix_shuffle_hid.py
    @desc : framework.model.shuffle_mix.
"""


import torch
import torch.nn as nn
from framework.model.shufflev2_mix.network.word_embedding import WordEmbedding
from framework.model.shufflev2_mix.network.hidden_mix import BalancedTransMix, SoftTransMix, HardTransMix


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


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


class ShuffleNetV2VQA(nn.Module):
    def __init__(self, in_chans=3, input_size=224, width_mult=1, qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="bal_hidden_mix", mix_probability=1, mix_alpha_1=5, mix_alpha_2=1):
        super(ShuffleNetV2VQA, self).__init__()

        assert input_size % 32 == 0

        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, 24)
        self.img_attn = MultiAttnFusion(24)

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(in_chans, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        self.qus_linear = nn.Sequential(nn.Linear(24, 1024),
                                        nn.LayerNorm(1024),
                                        nn.GELU())
        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], ans_size))

    def forward(self, x, y, label=None):
        y = self.qus_emb(y)

        x = self.conv1(x)
        x = self.maxpool(x)

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        x = self.img_attn(x, y)

        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = x.view(-1, self.stage_out_channels[-1])

        y_out = self.qus_linear(y).mean(1)

        out = self.classifier(x * y_out)

        if self.training:
            return out, mix_label
        else:
            return out


def mix_hid_shufflenetv2_base(**kwargs):
    model = ShuffleNetV2VQA(qus_embedding_dim=300, ans_size=kwargs["ans_size"],
                            glove_path=kwargs["glove_path"], word_size=kwargs["word_size"],
                            select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                            mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"]
                            )
    return model


if __name__ == "__main__":
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = ShuffleNetV2VQA().cuda()

    res1, res2 = model(a, b, c)
    print(res1.shape, res2.shape)
    print(sum(x.numel() for x in model.parameters()))  # 1615007







