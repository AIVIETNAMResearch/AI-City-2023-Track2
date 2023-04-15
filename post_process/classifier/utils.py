import copy
import json
import os
import os.path as osp
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class AverageTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def l2_loss():
    criterion = nn.MSELoss(reduction='sum')
    return criterion

def evaluate_tensor(y_pred, y_true, thres=0.5):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    y_pred = (y_pred >= thres).astype(np.int)

    return accuracy_score(y_true, y_pred)

def evaluate_one_hot(y_pred, y_true):
    y_pred = torch.softmax(y_pred,dim=-1)
    correct = (y_pred.argmax(dim=-1)==y_true).sum()
    return float(correct/y_pred.size(0))

def evaluate_fraction(y_pred, y_true, dist=1/3, thres=0.1, top_k=2):
    max_val = torch.max(y_pred, dim=1).values
    r, c = y_pred.shape
    min_val = torch.max(max_val - dist, torch.tensor([thres]*r).cuda())

    y_pred_idx = torch.topk(y_pred, k=top_k, dim=1).indices
    new_y_pred = torch.gather(y_pred, 1, y_pred_idx)
    new_y_pred = (new_y_pred >= torch.reshape(min_val, (r, -1))).float()

    r, c = new_y_pred.shape
    count_non_zero = c - (new_y_pred != 0).sum(dim=1)

    for i in range(len(y_pred_idx)):
        if count_non_zero[i]:
            y_pred_idx[i][-1] = y_pred_idx[i][0]

    new_y_true = torch.gather(y_true, 1, y_pred_idx)
    new_y_true = (new_y_true>0).float()

    return float((torch.sum(torch.mean(new_y_true, axis=-1))/r))

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, save_path=None,writer = None, one_hot=False):
    since = time.time()

    val_acc_history, train_acc_history = [], []
    val_loss_history, train_loss_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")
    step_count = 0

    trackers = {
        'train': AverageTracker(), 'val': AverageTracker()
    }
    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            trackers[phase].reset()

            loader = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            for it, data in loader:
                inputs = data['img'].cuda()
                labels = data['label'].cuda()

                batch_size = inputs.shape[0]
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(f'outputs shape: {outputs.shape}')
                    # print(f'labels shape: {labels.shape}')
                    try:
                        loss = criterion(outputs, labels)
                    except:
                        import pdb;pdb.set_trace();
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                running_loss += loss.item() * inputs.size(0)
                if one_hot:
                    it_score = evaluate_one_hot(outputs,labels)
                else:
                    it_score = evaluate_fraction(outputs, labels)
                trackers[phase].update(it_score, n=batch_size)
            # Print after each epoch
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = trackers[phase].avg
            if writer is not None:
                writer.add_scalar("train loss" if phase == 'train' else "val loss",epoch_loss,step_count)
                writer.add_scalar("val acc" if phase == 'train' else "val acc",epoch_loss,step_count)

            print('[{}, epoch {}/{}] Loss: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch, num_epochs, epoch_loss, epoch_acc)
            )

            # deep copy the model
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if save_path is not None:
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path),exist_ok=True)
                        torch.save(model.state_dict(), save_path)
                        print(f"save best model at {epoch}")
                    else:
                        step_count += 1
                    # pass the validation loss to the scheduler to judge whether to
                    # adjust learning rate 
                    scheduler.step(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

    
    # torch.save(model.state_dict(), osp.join(save_path+f'last_model.pt'))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

