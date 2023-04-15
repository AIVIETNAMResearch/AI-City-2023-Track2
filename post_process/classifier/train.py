import gc
import imp
import json
import os
import os.path as osp
from glob import glob
from tkinter.tix import Tree

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
# from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from box_extractor import init_model
from config import cfg_col, cfg_dir, cfg_veh
from dataset import (COL_BOX_DIR, COL_GROUP_JSON, COL_TRAIN_CSV, DIR_BOX_DIR,
                     DIR_TRAIN_CSV, VEH_BOX_DIR, VEH_GROUP_JSON, VEH_TRAIN_CSV,
                     VehicleDataset, get_dataset)
from utils import evaluate_fraction, evaluate_tensor, l2_loss, train_model

gc.collect()
torch.cuda.empty_cache()


veh_model, col_model, dir_model = init_model(cfg_veh, cfg_col, cfg_dir, load_ckpt=False)
veh_model = veh_model.cuda()
col_model = col_model.cuda()
dir_model = dir_model.cuda()

def train_model_type(model, cfg, csv_path: str, json_path: str, box_dir: str,writer, one_hot=False):
    df_train, df_val = get_dataset(csv_path, json_path, box_dir)

    train_dataset = VehicleDataset(df_train, 'train')
    val_dataset = VehicleDataset(df_val, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=2)

    # test 
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    sample = train_dataset[0]
    for k in sample.keys():
        print(f'{k} shape: {sample[k].shape}')

    if one_hot:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criteriion = l2_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-07, eps=1e-07, verbose=True)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    model, val_acc, train_acc = train_model(
        model, dataloaders, 
        criterion, optimizer, lr_scheduler, 
        num_epochs=cfg['train']['num_epochs'], 
        save_path=cfg['WEIGHT'],writer=writer,
        one_hot=one_hot
    )
    pass

def train_vehicle():
    print(f'TRAIN VEHICLE')
    log_dir = "./logs/vehicle_type"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir)
    writer = None
    train_model_type(veh_model, cfg_veh, VEH_TRAIN_CSV, VEH_GROUP_JSON, VEH_BOX_DIR, writer,one_hot=True)
    pass

def train_color():
    print(f'TRAIN COLOR')
    log_dir = "./logs/vehicle_color"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir)
    writer = None
    train_model_type(col_model, cfg_col, COL_TRAIN_CSV, COL_GROUP_JSON, COL_BOX_DIR,writer,one_hot=True)
    pass

def train_direction():
    print(f'TRAIN DIRECTION')
    log_dir = "./logs/vehicle_direction"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir)
    writer = None
    train_model_type(dir_model, cfg_dir, DIR_TRAIN_CSV, None, DIR_BOX_DIR,writer,one_hot=True)
    pass

def main():
    try:
        train_vehicle()
    except:
        import pdb;pdb.set_trace();
    try:
        train_color()
    except:
        import pdb;pdb.set_trace();
    try:
        train_direction()
    except:
        import pdb;pdb.set_trace();

if __name__ == '__main__':
    main()
    pass
