import torch
import torch.nn as nn
import timm
from collections import OrderedDict


def resnest50d(): 
    net = timm.create_model("resnest50d", pretrained=True)
    resnest50d_backbone = nn.Sequential(OrderedDict({
        "conv1": net.conv1,
        "bn1": net.bn1,
        "act1": net.act1,
        "maxpool": net.maxpool,
        "layer1": net.layer1,
        "layer2": net.layer2,
        "layer3": net.layer3,
        "layer4": net.layer4,
        "global_pool": net.global_pool,
    }))
    return resnest50d_backbone


