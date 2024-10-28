# _*_ coding: utf-8 _*_

"""
    @Time : 2023/4/22 20:47 
    @Author : smile 笑
    @File : __init__.py
    @desc :
"""


from .model_interface import ModelInterfaceModule


def get_model_module(model_name):
    if model_name == "general_densenet121_base":
        from .model.dense_mix.dense_net import general_densenet121
        return general_densenet121
    if model_name == "general_densenet169_base":
        from .model.dense_mix.dense_net import general_densenet169
        return general_densenet169
    if model_name == "general_densenet201_base":
        from .model.dense_mix.dense_net import general_densenet201
        return general_densenet201

    if model_name == "mix_hid_densenet121_base":
        from .model.dense_mix.mix_dense import mix_hid_densenet121
        return mix_hid_densenet121
    if model_name == "mix_hid_densenet169_base":
        from .model.dense_mix.mix_dense import mix_hid_densenet169
        return mix_hid_densenet169
    if model_name == "mix_hid_densenet201_base":
        from .model.dense_mix.mix_dense import mix_hid_densenet201
        return mix_hid_densenet201

    if model_name == "general_googlenet_base":
        from .model.google_mix.google_net import general_googlenet
        return general_googlenet
    if model_name == "mix_hid_googlenet_base":
        from .model.google_mix.mix_google import mix_hid_googlenet
        return mix_hid_googlenet

    if model_name == "general_mobilenet_base":
        from .model.mobile_mix.mobile_v1_net import general_mobilenet
        return general_mobilenet
    if model_name == "mix_hid_mobilenet_base":
        from .model.mobile_mix.mix_mobile_v1 import mix_hid_mobilenet
        return mix_hid_mobilenet
    if model_name == "mix_hid_mobilenetv2_base":
        from .model.mobile_mix.mix_mobile_v2 import mix_hid_mobilenetv2
        return mix_hid_mobilenetv2
    if model_name == "general_mobilenetv2_base":
        from .model.mobile_mix.mobile_v2_net import general_mobilenetv2
        return general_mobilenetv2
    if model_name == "general_mobilev3_base":
        from .model.mobilev3_mix.mobile_v3 import general_mobilev3_base
        return general_mobilev3_base
    if model_name == "mix_hid_mobilev3_base":
        from .model.mobilev3_mix.mix_mobile_hid import mix_hid_mobilev3_base
        return mix_hid_mobilev3_base

    if model_name == "general_resnet34_base":
        from .model.resnet_mix.resnet import general_resnet34_base
        return general_resnet34_base
    if model_name == "general_resnet50_base":
        from .model.resnet_mix.resnet import general_resnet50_base
        return general_resnet50_base
    if model_name == "general_resnet101_base":
        from .model.resnet_mix.resnet import general_resnet101_base
        return general_resnet101_base
    if model_name == "mix_hid_resnet34_base":
        from .model.resnet_mix.mix_resnet_hid import mix_hid_resnet34_base
        return mix_hid_resnet34_base
    if model_name == "mix_hid_resnet50_base":
        from .model.resnet_mix.mix_resnet_hid import mix_hid_resnet50_base
        return mix_hid_resnet50_base
    if model_name == "mix_hid_resnet101_base":
        from .model.resnet_mix.mix_resnet_hid import mix_hid_resnet101_base
        return mix_hid_resnet101_base

    if model_name == "general_shufflenet_base":
        from .model.shuffle_mix.shuffle_v1 import general_shufflenet
        return general_shufflenet
    if model_name == "mix_hid_shufflenet_base":
        from .model.shuffle_mix.mix_shuffle_v1 import mix_hid_shufflenet
        return mix_hid_shufflenet
    if model_name == "general_shufflenetv2_base":
        from .model.shufflev2_mix.shuffle_v2 import general_shufflenetv2_base
        return general_shufflenetv2_base
    if model_name == "mix_hid_shufflenetv2_base":
        from .model.shufflev2_mix.mix_shuffle_hid import mix_hid_shufflenetv2_base
        return mix_hid_shufflenetv2_base

    if model_name == "general_vgg13_bn_base":
        from .model.vgg_mix.vgg_net import general_vgg13_bn
        return general_vgg13_bn
    if model_name == "general_vgg16_bn_base":
        from .model.vgg_mix.vgg_net import general_vgg16_bn
        return general_vgg16_bn
    if model_name == "general_vgg19_bn_base":
        from .model.vgg_mix.vgg_net import general_vgg19_bn
        return general_vgg19_bn
    if model_name == "mix_hid_vgg13_bn_base":
        from .model.vgg_mix.mix_vgg_net import mix_hid_vgg13_bn
        return mix_hid_vgg13_bn
    if model_name == "mix_hid_vgg16_bn_base":
        from .model.vgg_mix.mix_vgg_net import mix_hid_vgg16_bn
        return mix_hid_vgg16_bn
    if model_name == "mix_hid_vgg19_bn_base":
        from .model.vgg_mix.mix_vgg_net import mix_hid_vgg19_bn
        return mix_hid_vgg19_bn

    if model_name == "general_efficient_b3_base":
        from .model.efficient_mix.efficient_v1_net import general_efficient_b3
        return general_efficient_b3
    if model_name == "mix_hid_efficient_b3_base":
        from .model.efficient_mix.efficient_v1_mix import mix_hid_efficient_b3
        return mix_hid_efficient_b3

    if model_name == "general_efficientv2_b3_base":
        from .model.efficient_mix.efficient_v2_net import general_efficientv2_b3
        return general_efficientv2_b3
    if model_name == "mix_hid_efficientv2_b3_base":
        from .model.efficient_mix.efficient_v2_mix import mix_hid_efficientv2_b3
        return mix_hid_efficientv2_b3

    if model_name == "general_regnetx_080_base":
        from .model.regnet_mix.regnet import general_regnetx_080
        return general_regnetx_080
    if model_name == "mix_hid_regnetx_080_base":
        from .model.regnet_mix.mix_regnet import mix_hid_regnetx_080
        return mix_hid_regnetx_080

    if model_name == "general_ghost_net_base":
        from .model.ghost_mix.ghost_net import general_ghost_net
        return general_ghost_net
    if model_name == "mix_hid_ghost_net_base":
        from .model.ghost_mix.mix_ghost import mix_hid_ghost_net
        return mix_hid_ghost_net

    if model_name == "general_convnext_base":
        from .model.convnext_mix.convnext import general_convnext_base
        return general_convnext_base
    if model_name == "mix_hid_convnext_base":
        from .model.convnext_mix.mix_convnext import mix_hid_convnext_base
        return mix_hid_convnext_base

    if model_name == "general_convnext_small":
        from .model.convnext_mix.convnext import general_convnext_small
        return general_convnext_small
    if model_name == "mix_hid_convnext_small":
        from .model.convnext_mix.mix_convnext import mix_hid_convnext_small
        return mix_hid_convnext_small

    if model_name == "general_edgenext_base":
        from .model.edgenext_mix.edgenext import general_edgenext_base
        return general_edgenext_base
    if model_name == "mix_hid_edgenext_base":
        from .model.edgenext_mix.mix_edgenext import mix_hid_edgenext_base
        return mix_hid_edgenext_base

    if model_name == "general_fasternet_m_base":
        from .model.faster_mix.faster_net import general_fasternet_m
        return general_fasternet_m
    if model_name == "mix_hid_fasternet_m_base":
        from .model.faster_mix.mix_faster import mix_hid_fasternet_m
        return mix_hid_fasternet_m

    if model_name == "general_fasternet_s_base":
        from .model.faster_mix.faster_net import general_fasternet_s
        return general_fasternet_s
    if model_name == "mix_hid_fasternet_s_base":
        from .model.faster_mix.mix_faster import mix_hid_fasternet_s
        return mix_hid_fasternet_s



