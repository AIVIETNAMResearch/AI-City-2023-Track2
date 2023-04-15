import numpy as np
import torch
from scipy.stats import gmean


def compute_retrieval_rank_train(simmat):    
    rank_res = (-simmat).argsort().argsort()
    ind = torch.diag(rank_res).cpu().numpy()
    return ind 


def compute_metric_from_rank_train(ind):
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics['geometric_mean_R1-R5-R10'] = gmean([metrics['R1'], metrics['R5'], metrics['R10']])
    metrics['MRR'] = np.mean(1./(ind + 1)) * 100
    return metrics


def retrieval_metric_train(simmat): 
    ind = compute_retrieval_rank_train(simmat)
    metrics = compute_metric_from_rank_train(ind)
    return metrics

def classification_metric(pred,gt):
    pred_idx = pred.argmax(dim=-1)
    correct = torch.sum(pred_idx==gt)
    count = gt.size(0)
    metrics = {}
    metrics['Acc'] = float(correct)/count * 100
    return metrics
