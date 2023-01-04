import torch
from torch import nn
from torch.nn import functional as F


class CosFacePairLoss(nn.Module):
    def __init__(self, s=30, m=0.50):
        super(CosFacePairLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, embedding, targets):
        embedding = F.normalize(embedding, p=2, dim=1)
        # if torch.distributed.is_initialized():
        #     embedding = AllGather(embedding)
        #     targets = AllGather(targets)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -self.s * s_p + (-99999999.0) * (1 - is_pos)
        logit_n = self.s * (s_n + self.m) + (-99999999.0) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss


class CosFacePairLoss_v2(nn.Module):
    def __init__(self, s=30, m_max=0.6, m_min=0.2, m=0):
        super(CosFacePairLoss_v2, self).__init__()
        self.s = s
        self.m_max = m_max
        self.m_min = m_min

    def forward(self, embedding, targets):
        embedding = F.normalize(embedding, p=2, dim=1)
        # if torch.distributed.is_initialized():
        #     embedding = AllGather(embedding)
        #     targets = AllGather(targets)

        dist_mat = torch.matmul(embedding, embedding.t())
        N = dist_mat.size(0)
        mask = is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
        p_simi = (dist_mat * is_pos).sum(1) / is_pos.sum(1)
        p_simi = p_simi.mean()
        m = (self.m_max - self.m_min) * p_simi + self.m_min
        mask_hard_neg = ((p_simi - m) < dist_mat) * (1 - mask)
        scale_matrix = self.s * (1 - mask_hard_neg) + self.s * mask_hard_neg * (p_simi + self.m_max - self.m_min)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -scale_matrix * s_p + (-99999999.0) * (1 - is_pos)
        logit_n = scale_matrix * (s_n + m) + (-99999999.0) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss
    

class CirclePairLoss(nn.Module):
    def __init__(self, s=30, m=0.30):
        super(CirclePairLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, embedding, targets):
        embedding = F.normalize(embedding, p=2, dim=1)
        # if torch.distributed.is_initialized():
        #     embedding = AllGather(embedding)
        #     targets = AllGather(targets)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.s * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = self.s * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss
