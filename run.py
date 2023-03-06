import json
import os
import pandas as pd
from sklearn.model_selection import KFold
import random
import numpy as np
import torch
from dotwiz import DotWiz
from trainer.train import train_loop
from datetime import datetime
import logging
logging.basicConfig(filename='log/training.log', filemode='w')
from transformers import AutoTokenizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

def load_training_data(config):
    with open(os.path.join(config['general_config']["data_dir"], "train_nl_extracted_.json"), "r") as f:
        data = json.load(f)

    data_df = pd.DataFrame(data).transpose().reset_index()
    data_df = data_df.rename(columns={'index': 'uuid'})
    data_df['colors'] = data_df['colors'].replace(['white', 'black', 'gray', 'red', 'blue', 'green', 'brown', 'yellow'], range(8))
    data_df['type'] = data_df['type'].replace(['sedan', 'truck', 'suv', 'van', 'bus', 'hatchback'], range(6))
    data_df['motion'] = data_df['motion'].replace(['straight', 'left', 'right', 'stop'], range(4))

    num_folds = config['general_config']['kfolds']
    kfolds = KFold(n_splits=num_folds, shuffle=True)

    data_df['fold'] = -1
    for fold, (train_index, test_index) in enumerate(kfolds.split(data_df.index)):
        data_df.loc[test_index, ['fold']] = fold

    return data_df


def main():

    with open('./configs/baseline_config.json', "r") as f:
        config = json.load(f)
    print(config)

    config = DotWiz(config)
    
    # logging.info(config)
    tokenizer = AutoTokenizer.from_pretrained(config['arch']['base_settings']['text_params']['model'], use_fast=True)

    config['general_config']['tokenizer'] = tokenizer

    data_df = load_training_data(config)

    for fold in [3,4]:
        train_fold = data_df[data_df['fold'] != fold].reset_index(drop=True)
        valid_fold = data_df[data_df['fold'] == fold].reset_index(drop=True)
        
        model_checkpoint_path = f'./checkpoint'
        print(f"TRAINING FOLD {fold + 1}")
        train_loop(train_fold,
                valid_fold,
                fold=fold,
                model_checkpoint_path=model_checkpoint_path,
                device=device,
                config=config)

if __name__ == "__main__":
    main()