import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from box_extractor import init_model, preprocess_input
from config import cfg_col, cfg_dir, cfg_veh

TRACKS_DIR = "/root/paddlejob/workspace/env_run/output/zhangjc/data/aicity/annotations/relation_info_front_test.json"
track = json.load(open(TRACKS_DIR))
save_dir = "./results"
scale = 1.2
def label_extraction(model, track, num_classes, sparse=False, avg=True, one_hot=True):
    ans = {}
    split = 10
    for key in tqdm(track):

        l = len(track[key]["boxes"])
        idx = range(l)
        total_boxes = track[key]['boxes']
        frames = [track[key]['frames'][i] for i in idx if total_boxes[i]!= -1] 
        boxes = [track[key]['boxes'][i] for i in idx if total_boxes[i]!= -1]
        mask = [1 if total_boxes[i]!=-1 else 0 for i in idx]

        # check if there is no back or front car
        if len(frames)==0:
            pred_label = [-1]*num_classes
            ans[key] = pred_label
            continue
        
        # sparse sample for prediction or not
        if sparse:
            l = len(boxes)
            idx = split_data(range(l), min(split,l))
            frames = [frames[i] for i in idx]
            boxes = [boxes[i] for i in idx]

        # predict every frame
        pred = []
        for frame, box in zip(frames, boxes):
            new_path = os.path.join("/root/paddlejob/workspace/env_run/output/zhangjc/data/aicity/",frame)
            img = Image.open(new_path)
            # pay attention to the box coordinate format
            x1, y1, x2, y2 = box
            x,y,w,h = x1,y1,x2-x1,y2-y1
            box = (int(x-(scale-1)*w/2.),int(y-(scale-1)*h/2),int(x+(scale+1)*w/2.),int(y+(scale+1)*h/2.))
            x_0, y_0, x_1, y_1 = box
            crop_img = img.crop((x_0, y_0, x_1, y_1))
            crop_img = preprocess_input(crop_img).cuda()
            crop_img = crop_img.unsqueeze(0)
            with torch.no_grad():
                pred.append(torch.softmax(model(crop_img),dim=-1))
        # check if only save average probability or all probability
        if avg:
            # keep only one probability for one track
            pred = torch.mean(torch.stack(pred), dim=0).cpu().detach()
            if one_hot:
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
            pred_label = []
            ind = 0
            for i in range(len(track[key]["boxes"])):
                if mask[i]:
                    pred_label.append(pred[ind].squeeze().cpu().detach().numpy().tolist())
                    ind += 1
                else:
                    pred_label.append(-1)
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

    veh_model = veh_model.cuda()
    col_model = col_model.cuda()

    veh_model.eval()
    col_model.eval()

    # predict vehicle label for test track's visual data
    ans_veh = label_extraction(veh_model, track, 6, sparse=False, avg=False, one_hot=False)
    save_dir_veh = os.path.join(save_dir, "front_test_vehicle_predict_no_avg.json")
    save_result(save_dir_veh, ans_veh)

    # predict color label for test track's visual data
    col_veh = label_extraction(col_model, track, 8, sparse=False, avg=False, one_hot=False)
    save_dir_col = os.path.join(save_dir, "front_test_color_predict_no_avg.json")
    save_result(save_dir_col, col_veh)
