import json
import torch
import os
import random
import numpy as np
import pandas as pd
from dataloader.datasets import get_infer_dataloader
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from trainer.utils import get_model
from dotwiz import DotWiz
from tqdm import tqdm
from trainer.utils import sim_matrix
from transformers import AutoTokenizer
import copy

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

def soup_models(model_list):
    souped_model = copy.deepcopy(model_list[0])
    for param in souped_model.named_parameters():
        name = param[0]
        
        model_params = model_list[0].state_dict()[name]
        for model in model_list[1:]:
            model_params += model.state_dict()[name]

        model_params = model_params / len(model_list)

        param[1].data = model_params

    return souped_model

def main():
    with open('./configs/baseline_config.json', "r") as f:
        config = json.load(f)
    with open("./data/AIC23_Track2_NL_Retrieval/data/test-queries.json", "r") as f:
        test_querries = json.load(f)
    with open("./data/AIC23_Track2_NL_Retrieval/data/test-tracks.json", "r") as f:
        test_tracks = json.load(f)

    config = DotWiz(config)

    tokenizer = AutoTokenizer.from_pretrained(config['arch']['base_settings']['text_params']['model'], use_fast=True)

    config['general_config']['tokenizer'] = tokenizer

    text_df = pd.DataFrame(test_querries).transpose().reset_index()
    text_df = text_df.rename(columns={'index': 'uuid'})

    video_df = pd.DataFrame(test_tracks).transpose().reset_index()
    video_df = video_df.rename(columns={'index': 'uuid'})

    video_dataloader, text_dataloader = get_infer_dataloader(config, video_df, text_df)

    # checkpoint_list = [
    #     'checkpoint/model_ckpt-fold0.pth',
    #     'checkpoint/model_ckpt-fold1.pth',
    #     'checkpoint/model_ckpt-fold2.pth',
    #     'checkpoint/model_ckpt-fold3.pth',
    #     'checkpoint/model_ckpt-fold0.pth'
    # ]

    # model_list = []

    # for ckpt in checkpoint_list:
    #     model = get_model(config, model_checkpoint_path=ckpt)
    #     model_list.append(model)

    # model = soup_models(model_list)
    model = get_model(config, model_checkpoint_path='checkpoint/model_ckpt-fold0.pth')
    model.to(device)
    model.eval()

    video_embeddings = []
    text_embeddings = []

    print("Compute video embeddings")
    for idx, inputs in tqdm(enumerate(video_dataloader), total=len(video_dataloader)):
        for k, v in inputs.items():
            if not isinstance(v, dict):
                inputs[k] = v.to(device)
                continue
            for k_, v_ in v.items():
                if(isinstance(v_, list)):
                    inputs[k][k_] = [val.to(device) for val in v_]
                elif(isinstance(v_, dict)):
                    for k__, v__ in v_.items():
                        inputs[k][k_][k__] = v__.to(device)
                else:
                    inputs[k][k_] = v_.to(device)

        with torch.no_grad():
            video_embs, _, _, _ = model.compute_video(inputs['video'], inputs['motion'], inputs['motion_line']) 
             #print(video_embs.shape)
            video_embeddings.append(video_embs)
    
    print("Compute text embeddings")
    batch_size = config.general_config.valid_batch_size
    for idx, inputs in tqdm(enumerate(text_dataloader), total=len(text_dataloader)):
        for k, v in inputs.items():
            if not isinstance(v, dict):
                inputs[k] = v.to(device)
                continue
            for k_, v_ in v.items():
                if(isinstance(v_, list)):
                    inputs[k][k_] = [val.to(device) for val in v_]
                elif(isinstance(v_, dict)):
                    for k__, v__ in v_.items():
                        inputs[k][k_][k__] = v__.to(device)
                else:
                    inputs[k][k_] = v_.to(device)
        
        with torch.no_grad():

            # last batch
            if idx == len(video_dataloader) - 1:
                batch_size = len(text_df) % config.general_config.valid_batch_size

            text_embs = model.compute_text(inputs['text'], batch_size) 
            #print(text_embs.shape)

            text_embeddings.append(text_embs)

    text_embeddings = torch.concat(text_embeddings)
    video_embeddings = torch.concat(video_embeddings)

    #print(text_embeddings.shape)
    #print(video_embeddings.shape)

    sim = sim_matrix(text_embeddings, video_embeddings)
    rank_matrix = torch.argsort(-1*sim)
    rank_matrix = rank_matrix.cpu().numpy()
    submission = {}
    #print(rank_matrix.shape)

    for idx, item in enumerate(text_df.values):
        uuid = item[0]
        sample_rank = rank_matrix[idx]

        video_rank = video_df.iloc[sample_rank]
        uuid_rank = [vid[0] for vid in video_rank.values]
        submission[uuid] = uuid_rank

    with open("submissions/submission2.json", "w") as f:
        json.dump(submission, f)

if __name__ == "__main__":
    main()
        

    

            



