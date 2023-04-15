import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from box_extractor import init_model, preprocess_input
from config import cfg_col, cfg_dir, cfg_veh

TRACKS_DIR = '../data/aicity/annotations/test_tracks.json'
track = json.load(open(TRACKS_DIR))

img_dir = "../srl_handler/results/veh_dir_imgs"
save_dir = "./results"

def motion_label_extraction(model, track, one_hot=True):
    ans = {}
    for key in tqdm(track):
       
        pred = []
        for i in range(4):
            img_path = os.path.join(img_dir, "motion_lines_{}/{}.jpg".format(i+1,key))
            img = Image.open(img_path)
           
            input_img = preprocess_input(img).cuda()
            input_img = input_img.unsqueeze(0)
            with torch.no_grad():
                pred.append(torch.softmax(model(input_img),dim=-1))
        pred = torch.mean(torch.stack(pred), dim=0).cpu().detach()
        print(pred.shape)
        if one_hot:
            # change to the 0, 1 label 
            ind = pred[0].argmax()
            pred_label = [0]*pred.size(1)
            pred_label[ind] = 1

            ans[key] = pred_label
        else:
            pred_label = pred[0].numpy().tolist()
            ans[key] = pred_label
    return ans

def split_data(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    ans = []
    for arr in out:
        ans.append(arr[len(arr)//2])
    return ans

def save_result(save_dir, data_json):
    f = open(save_dir, 'w')
    json.dump(data_json, f, indent=2)
    f.close()

if __name__ == "__main__":
    veh_model, col_model, dir_model = init_model(cfg_veh, cfg_col, cfg_dir, load_ckpt=True)
    dir_model.cuda()
    dir_model.eval()

    # predict direction label for test track's visual data
    ans_veh = motion_label_extraction(dir_model, track, one_hot=True)
    save_dir_veh = os.path.join(save_dir, "target_test_direction_predict_ont_hot.json")
    save_result(save_dir_veh, ans_veh)

    # ans_veh = motion_label_extraction(dir_model,track,one_hot=False)
    # save_dir_veh = os.path.join(save_dir,"target_test_direction_predict_softmax.json")
    # save_result(save_dir_veh, ans_veh)

    
