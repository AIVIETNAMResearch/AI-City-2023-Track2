"""
@author:  chenhaobo
@contact: hbchen121@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(inputs, targets, epsilon=0.):
    num_classes = inputs.size(1)
    log_probs = F.log_softmax(inputs, dim=1)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=0, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        # self.num_classes = num_classes
        if epsilon <= 0:
            epsilon = 0.
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        num_classes = inputs.size(1)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
