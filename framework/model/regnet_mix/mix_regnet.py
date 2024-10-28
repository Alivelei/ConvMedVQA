# _*_ coding: utf-8 _*_

"""
    @Time : 2023/6/22 10:39
    @Author : smile 笑
    @File : regnet.py
    @desc : framework.model.regnet_mix.
"""


import torch
import torch.nn as nn
from framework.model.regnet_mix.network.word_embedding import WordEmbedding
from framework.model.regnet_mix.network.hidden_mix import BalancedTransMix, SoftTransMix, HardTransMix


__all__ = ['regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032',
           'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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


class RegNet(nn.Module):
    def __init__(self, block, layers, widths, zero_init_residual=True,
                 group_width=1, replace_stride_with_dilation=None, norm_layer=None, qus_embedding_dim=300,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", word_size=305, ans_size=223,
                 select_mix_flag="bal_hidden_mix", mix_probability=1, mix_alpha_1=5, mix_alpha_2=1):
        super(RegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        self.conv1 = self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.qus_emb = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, 64)
        self.img_attn = MultiAttnFusion(64)

        if select_mix_flag == "hard_hidden_mix":
            self.hidden_mix = HardTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "soft_hidden_mix":
            self.hidden_mix = SoftTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)
        elif select_mix_flag == "bal_hidden_mix":
            self.hidden_mix = BalancedTransMix(mix_alpha_1, mix_alpha_2, mix_probability, num_classes=ans_size)

        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.qus_linear = nn.Sequential(nn.Linear(64, widths[-1] * block.expansion),
                                        nn.LayerNorm(widths[-1] * block.expansion),
                                        nn.GELU())

        self.fc = nn.Linear(widths[-1] * block.expansion, ans_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, y, label=None):
        # See note [TorchScript super()]
        x = self.conv1(x)
        y = self.qus_emb(y)

        if self.training:
            mix_label, _ = self.hidden_mix(x, y, label)

        x = self.img_attn(x, y)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        y_out = self.qus_linear(y).mean(1)
        out = self.fc(x * y_out)

        if self.training:
            return out, mix_label
        else:
            return out


# 2618203
def mix_hid_regnetx_002(**kwargs):
    return RegNet(Bottleneck, [1, 1, 4, 7], [24, 56, 152, 368], group_width=8, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 5080075
def mix_hid_regnetx_004(**kwargs):
    return RegNet(Bottleneck, [1, 2, 7, 12], [32, 64, 160, 384], group_width=16, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 6017387
def mix_hid_regnetx_006(**kwargs):
    return RegNet(Bottleneck, [1, 3, 5, 7], [48, 96, 240, 528], group_width=24, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 6979787
def mix_hid_regnetx_008(**kwargs):
    return RegNet(Bottleneck, [1, 3, 7, 5], [64, 128, 288, 672], group_width=16, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 8740379
def mix_hid_regnetx_016(**kwargs):
    return RegNet(Bottleneck, [2, 4, 10, 2], [72, 168, 408, 912], group_width=24, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 14780171
def mix_hid_regnetx_032(**kwargs):
    return RegNet(Bottleneck, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 21350923
def mix_hid_regnetx_040(**kwargs):
    return RegNet(Bottleneck, [2, 5, 14, 2], [80, 240, 560, 1360], group_width=40, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 25260123
def mix_hid_regnetx_064(**kwargs):
    return RegNet(Bottleneck, [2, 4, 10, 1], [168, 392, 784, 1624], group_width=56, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 38407723
def mix_hid_regnetx_080(**kwargs):
    return RegNet(Bottleneck, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 44723147
def mix_hid_regnetx_120(**kwargs):
    return RegNet(Bottleneck, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 53033995
def mix_hid_regnetx_160(**kwargs):
    return RegNet(Bottleneck, [2, 6, 13, 1], [256, 512, 896, 2048], group_width=128, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


# 106237019
def mix_hid_regnetx_320(**kwargs):
    return RegNet(Bottleneck, [2, 7, 13, 1], [336, 672, 1344, 2520], group_width=168, qus_embedding_dim=300,
                  glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
                  select_mix_flag=kwargs["select_mix_flag"], mix_probability=kwargs["mix_probability"],
                  mix_alpha_1=kwargs["mix_alpha_1"], mix_alpha_2=kwargs["mix_alpha_2"])


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    c = torch.randint(0, 100, [2]).cuda()

    model = RegNet(Bottleneck, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120).cuda()

    res, m_l = model(a, b, c)
    print(res.shape, m_l.shape)
    print(sum(x.numel() for x in model.parameters()))

