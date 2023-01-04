import numpy as np
import torch
import time
import json
from utils import AverageMeter, ProgressMeter, get_mrr, accuracy
from tqdm import tqdm
import torch.nn.functional as F
from datetime import timedelta, datetime


def rerank_with_params(all_lang_embeds, all_visual_embeds, params, rank_type='k-rec'):
    k1, k2, lamb = params
    if rank_type == 'k-rec':
        # k-reciprocal re-rank
        query_feature = []
        gallery_feature = []
        for track_id in sorted(all_visual_embeds.keys()):
            lang_embed = all_lang_embeds[track_id]
            visual_embed = all_visual_embeds[track_id]
            gallery_feature.append(visual_embed.cpu().numpy())
            query_feature.append(lang_embed.cpu().numpy())

        gallery_feature = np.array(gallery_feature)
        query_feature = np.array(query_feature)

        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
        sim_rerank = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lamb)
        sim_rerank = torch.from_numpy(sim_rerank)
        mrr_ = get_mrr(sim_rerank)
    else:
        # gnn re-rank
        query_feature = []
        gallery_feature = []
        for track_id in sorted(all_visual_embeds.keys()):
            lang_embed = all_lang_embeds[track_id]
            visual_embed = all_visual_embeds[track_id]
            gallery_feature.append(visual_embed)
            query_feature.append(lang_embed)

        gallery_feature = torch.stack(gallery_feature, dim=0)
        query_feature = torch.stack(query_feature, dim=0)

        ori_dist, qfeat_new, gfeat_new = GNN_rerank(query_feature, gallery_feature, k1=k1, k2=k2, eps=lamb)
        sim_rerank = torch.matmul(qfeat_new, gfeat_new.T).cpu().numpy()
        mrr_ = get_mrr(sim_rerank)

    return mrr_.item()*100, params, sim_rerank


def gnn_rerank_params_grid_search(all_lang_embeds, all_visual_embeds):
    query_feature = []
    gallery_feature = []
    for track_id in sorted(all_visual_embeds.keys()):
        lang_embed = all_lang_embeds[track_id]
        visual_embed = all_visual_embeds[track_id]
        gallery_feature.append(visual_embed)
        query_feature.append(lang_embed)

    gallery_feature = torch.stack(gallery_feature, dim=0)
    query_feature = torch.stack(query_feature, dim=0)

    # query_feature = torch.sum(torch.cat(query_feature, dim=0) , dim=1, keepdim=False)

    print("====> gnn rerank grid search ...")
    # k1_params = [10, 15, 20, 25, 30, 35]
    k2_params = [3, 5, 7, 9]
    lambda_params = [0.3, 0.5, 0.7, 0.9]
    k1_params = range(10, 35, 2)
    # k2_params = range(1, 10, 1)
    # k2_params = range(1, 35, 1)
    # lambda_params = (0.1 * i for i in range(1, 10, 1))
    best_mrr = 0
    best_params = []
    best_sim_rerank = None
    import itertools
    for k1, k2, lamb in itertools.product(k1_params, k2_params, lambda_params):
        if k2 >= k1:
            continue
        ori_dist, qfeat_new, gfeat_new = GNN_rerank(query_feature, gallery_feature, k1=k1, k2=k2, eps=lamb)
        sim_rerank = torch.matmul(qfeat_new, gfeat_new.T)
        mrr_ = get_mrr(sim_rerank)
        # print(f"k1 = {k1}, k2 = {k2}, eps ={lamb}, mrr={mrr_.item()*100}")
        if mrr_.item() * 100 > best_mrr:
            best_mrr = mrr_.item() * 100
            best_params = (k1, k2, lamb)
            best_sim_rerank = sim_rerank
            rerank_acc1, rerank_acc5 = accuracy(sim_rerank, torch.arange(sim_rerank.size(0)).to(sim_rerank.device),
                                                topk=(1, 5))
            print(
                f"better_mrr = {best_mrr}, k1 = {k1}, k2 = {k2}, eps = {lamb}, acc1 = {rerank_acc1}, acc5 = {rerank_acc5}")
    return best_mrr, best_params, best_sim_rerank


def rerank_params_grid_search(all_lang_embeds, all_visual_embeds, rank_type='k-rec'):
    """
    :params: all_lang_embeds, dict, key为track_id, value为img_feature
    """
    if rank_type == 'gnn':
        return gnn_rerank_params_grid_search(all_lang_embeds, all_visual_embeds)
    query_feature = []
    gallery_feature = []
    for track_id in sorted(all_visual_embeds.keys()):
        lang_embed = all_lang_embeds[track_id]
        visual_embed = all_visual_embeds[track_id]
        gallery_feature.append(visual_embed.cpu().numpy())
        query_feature.append(lang_embed.cpu().numpy())

    gallery_feature = np.array(gallery_feature)
    query_feature = np.array(query_feature)
    # query_feature = np.sum(np.array(query_feature) , axis=1)

    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

    print("====> k-reciprocal rerank grid search ...")
    # k1_params = [10, 15, 20, 25, 30, 35]
    # k2_params = [3, 5, 7, 9]
    # lambda_params = [0.3, 0.5, 0.7, 0.9]
    k1_params = range(10, 35, 5)
    k2_params = range(1, 10, 2)
    # k2_params = range(1, 35, 1)
    lambda_params = (0.1 * i for i in range(1, 10, 2))
    best_mrr = 0
    best_params = []
    best_sim_rerank = None
    import itertools
    for k1, k2, lamb in itertools.product(k1_params, k2_params, lambda_params):
        if k2 >= k1:
            continue
        sim_rerank = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lamb)
        sim_rerank = torch.from_numpy(sim_rerank)
        mrr_ = get_mrr(sim_rerank)
        # print(f"k1={k1}, k2={k2}, eps={lamb}, mrr={mrr_.item()*100}")
        if mrr_.item()*100 > best_mrr:
            best_mrr = mrr_.item()*100
            best_params = (k1, k2, lamb)
            best_sim_rerank = sim_rerank
            rerank_acc1, rerank_acc5 = accuracy(sim_rerank, torch.arange(sim_rerank.size(0)).to(sim_rerank.device),
                                                topk=(1, 5))
            print(f"better_mrr = {best_mrr}, k1 = {k1}, k2 = {k2}, eps = {lamb}, acc1 = {rerank_acc1}, acc5 = {rerank_acc5}")
    return best_mrr, best_params, best_sim_rerank


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    """
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1), np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, : int(np.around(k1 / 2.0)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2.0)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2.0 / 3 * len(
                candidate_k_reciprocal_index
            ):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)
    original_dist = original_dist[
        :query_num,
    ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


@torch.no_grad()
def GNN_rerank(feat_q, feat_g, k1=26, k2=9, alpha=2, layers=2, eps=0., print_time=True, rerank_w=1, ori_w=0, **kwargs):
    """
        feat_q: query feature, shape (N,D) or (D,)
        feat_g: gallery feature, shape (M,D) or (D,)
        k1, k2: rerank hyperparams for k-reciprocal.
        layers: number of gnn layers.
        ori_w, rerank_w: weight for original and reranked distance matrix.
        eps: cosine similarity between 0 and 1
    """
    start_time = time.time()
    if isinstance(feat_q, (np.ndarray, list)):
        feat_q = torch.Tensor(feat_q)

    if isinstance(feat_g, (np.ndarray, list)):
        feat_g = torch.Tensor(feat_g)

    if feat_q.dim() == 1:
        feat_q = feat_q.unsqueeze(0)

    if feat_g.dim() == 1:
        feat_g = feat_g.unsqueeze(0)

    if torch.cuda.is_available():
        feat_q = feat_q.cuda(0)
        feat_g = feat_g.cuda(0)

    eps = torch.tensor(eps).to(feat_q.device)
    Q = feat_q.size(0)

    ori_simi = torch.mm(feat_q, feat_g.t())
    # ori_dist = 2 - 2 * ori_simi
    ori_dist = ori_simi

    features = torch.cat((feat_q, feat_g), dim=0)
    S = torch.mm(features, features.t())
    threshold = torch.topk(S, k1, sorted=False).values[:, -1].unsqueeze(-1)
    threshold = torch.maximum(threshold, eps)
    A = (S >= threshold).float()
    H = torch.nn.functional.normalize((A + A.t()) / 2, p=2, dim=1)

    threshold = torch.topk(S, k2, sorted=False).values[:, -1].unsqueeze(-1)
    threshold = torch.maximum(threshold, eps)
    S *= S >= threshold
    S = pow(S, alpha)

    for l in range(0, layers):
        H = torch.nn.functional.normalize(H + torch.mm(S, H), p=2, dim=1)
    # if print_time:
    #     print("time :", time.time() - start_time)

    # rerank_dist = 2 - 2 * torch.mm(H[:Q], H[Q:].t())
    rerank_dist = torch.mm(H[:Q], H[Q:].t())
    ori_dist = rerank_w * rerank_dist + ori_w * ori_dist

    qfeat_new = H[:Q]
    gfeat_new = H[Q:]
    if torch.cuda.is_available:
        ori_dist = ori_dist.cpu()
        qfeat_new = qfeat_new.cpu()
        gfeat_new = gfeat_new.cpu()
    # ori_dist = ori_dist.numpy()
    # qfeat_new = qfeat_new.numpy()
    # gfeat_new = gfeat_new.numpy()
    return ori_dist, qfeat_new, gfeat_new
