"""
@author:  chenhaobo
@contact: hbchen121@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


__all__ = [
    "Linear",
    "NormSoftmax",
    "ArcSoftmax",
    "CosSoftmax",
    "CircleSoftmax",
]


def cos_softmax(logits, targets, m):
    index = torch.where(targets != -1)[0]
    m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
    m_hot.scatter_(1, targets[index, None], m)
    logits[index] -= m_hot
    # # 由于 mem 计算使用多个 logits，共用时出错
    # logits = logits[index] - m_hot
    return logits


def arc_softmax(logits, targets, m):
    index = torch.where(targets != -1)[0]
    m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
    m_hot.scatter_(1, targets[index, None], m)
    logits.acos_()
    logits[index] += m_hot
    logits = logits.cos_()
    return logits


def circle_softmax(logits, targets, m):
    alpha_p = torch.clamp_min(-logits.detach() + 1 + m, min=0.)
    alpha_n = torch.clamp_min(logits.detach() + m, min=0.)
    delta_p = 1 - m
    delta_n = m

    # When use model parallel, there are some targets not in class centers of local rank
    index = torch.where(targets != -1)[0]
    m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
    m_hot.scatter_(1, targets[index, None], 1)

    logits_p = alpha_p * (logits - delta_p)
    logits_n = alpha_n * (logits - delta_n)

    logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

    neg_index = torch.where(targets == -1)[0]
    logits[neg_index] = logits_n[neg_index]
    return logits


class Linear(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=32, margin=0.35):
        super().__init__()
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        nn.init.normal_(self.weight, 0, 0.01)

    def forward(self, features, targets=None):
        return F.linear(features, self.weight)


class NormSoftmax(Linear):
    """ normFace """
    def forward(self, features, targets=None):
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        logits.mul_(self.s)
        return logits


class CosSoftmax(Linear):
    """ cosFace / AMsoftmax"""
    def forward(self, features, targets=None):
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        logits = cos_softmax(logits, targets, self.m)
        logits.mul_(self.s)
        return logits


class ArcSoftmax(Linear):
    """ ArcFace"""
    def forward(self, features, targets=None):
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        logits = arc_softmax(logits, targets, self.m)
        logits.mul_(self.s)
        return logits


class CircleSoftmax(Linear):
    """ Circle Loss"""
    def forward(self, features, targets=None):
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        logits = circle_softmax(logits, targets, self.m)
        logits.mul_(self.s)
        return logits
