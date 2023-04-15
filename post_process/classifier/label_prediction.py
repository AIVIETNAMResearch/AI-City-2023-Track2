import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from box_extractor import init_model, preprocess_input
from config import cfg_col, cfg_dir, cfg_veh

TRACKS_DIR = '../data/aicity/annotations/test_tracks.json'
track = json.load(open(TRACKS_DIR))
save_dir = "./results"

def label_extraction(model, track,sparse=True, avg=False, one_hot=True):
    ans = {}
    split = 10
    for key in tqdm(track):
        l = len(track[key]["boxes"])
        # sample some fixed num frames
        if sparse:
            idx = split_data(range(l), min(split,l))
        else:
            idx = range(l)
        frames = [track[key]['frames'][i] for i in idx]
        boxes = [track[key]['boxes'][i] for i in idx]
        pred = []
        for frame, box in zip(frames, boxes):
            new_path = os.path.join("../data/aicity",frame)
            img = Image.open(new_path)
            x, y, w, h = box
            x_0, y_0, x_1, y_1 = int(x), int(y), int(x+w), int(y+w)
            crop_img = img.crop((x_0, y_0, x_1, y_1))
            crop_img = preprocess_input(crop_img).cuda()
            crop_img = crop_img.unsqueeze(0)
            with torch.no_grad():
                pred.append(torch.softmax(model(crop_img),dim=-1))
        
        # check if only save average probability or all the probability
        if avg:
            pred = torch.mean(torch.stack(pred), dim=0).cpu().detach()
            if one_hot:
                print(pred.shape)
                # change to the 0, 1 label 
                ind = pred[0].argmax()
                pred_label = [0]*pred.size(1)
                pred_label[ind] = 1

                ans[key] = pred_label
            else:
                pred_label = pred[0].numpy().tolist()
                ans[key] = pred_label
        else:
            # keep every frame's probability
            pred_label = dict()
            for i,frame in enumerate(frames):
                pred_label[frame] = pred[i].squeeze().cpu().detach().numpy().tolist()

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
    veh_model, col_model ,dir_model= init_model(cfg_veh, cfg_col, cfg_dir,load_ckpt=True)

    veh_model = veh_model.cuda()
    col_model = col_model.cuda()

    veh_model.eval()
    col_model.eval()

    # predict vehicle label for test track's visual data
    ans_veh = label_extraction(veh_model, track, avg=True, one_hot=True)
    save_dir_veh = os.path.join(save_dir, "target_test_vehicle_predict_one_hot.json")
    save_result(save_dir_veh, ans_veh)

    # predict color label for test track's visual data
    col_veh = label_extraction(col_model, track, avg=True, one_hot=True)
    save_dir_col = os.path.join(save_dir, "target_test_color_predict_one_hot.json")
    save_result(save_dir_col, col_veh)
