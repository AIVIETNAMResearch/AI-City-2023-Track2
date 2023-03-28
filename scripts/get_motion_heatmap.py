import cv2
import json
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import multiprocessing
from functools import partial
import glob
import argparse
import IPython
import matplotlib.pyplot as plt

#from config import BASE_DIR
BASE_DIR = '/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Track2-Vehicle-Retrieval/Track2-Vehicle_Retrieval'

# dataset_path = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval'
# data_path = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval/new_baseline/AIC21_Track5_NL_Retrieval'

dataset_path = BASE_DIR + '/data/AIC23_Track2_NL_Retrieval/data'
data_path = BASE_DIR + '/data/AIC23_Track2_NL_Retrieval/data'

n_worker = 2
imgpath = dataset_path
# with open("data2021/test-tracks.json") as f:
#     tracks_test = json.load(f)
# with open("data2021/train-tracks.json") as f:
#     tracks_train = json.load(f)
with open("data/AIC23_Track2_NL_Retrieval/data/test-tracks.json") as f:
    tracks_test = json.load(f)
with open("data/AIC23_Track2_NL_Retrieval/data/train-tracks.json") as f:
    tracks_train = json.load(f)

save_heatmap_dir = osp.join(data_path, "data/motion_line")
os.makedirs(save_heatmap_dir, exist_ok=True)

all_tracks = tracks_train
for track in tracks_test:
    all_tracks[track] = tracks_test[track]


def get_motion_heatmap(data_dir, uuid, boxes_list):
    first = cv2.imread(os.path.join(data_dir, f'data/motion_map_iou/{uuid}.jpg'))
    w, h, c = first.shape
    line_motion_heatmap = first
    d = 100
    prev_point = []
    for idx, boxes in enumerate(boxes_list):
        x, y, w, h = boxes
        if len(prev_point) == 0:
            prev_point = [x, y, w, h]
        
        x_, y_, w_, h_ = prev_point
        point1 = (int(x_+w_/2), int(y_+h_/2))
        point2 = (int(x+ w/2),int(y + h/2))        
        cv2.line(line_motion_heatmap, point1, point2, [0,255,255], 10);

        # if np.mean(line_motion_heatmap[y:y+h, x:x+w]) < 50 or idx == len(boxes_list) - 1:
        #     color = np.mean(line_motion_heatmap[y:y+h, x:x+w]) + 1
        #     point1 = (int(x_+w_/2), int(y_+h_/2))
        #     point2 = (int(x+ w/2),int(y + h/2))
        #     if idx == len(boxes_list) - 1:
        #         displacement_vec = ((point2[0] - point1[0]), (point2[1] - point1[1]))
        #         eps = 1e-8
        #         coef = np.sqrt(d**2 / (displacement_vec[0]**2 + displacement_vec[1] ** 2 + eps))
        #         displacement_vec = (displacement_vec[0] * coef, displacement_vec[1] * coef)
        #         point2_ = (int(displacement_vec[0] + point1[0]), int(displacement_vec[1] + point1[1]))
        #         #print(point2_, " point2")
        #         #print(point1, " point1")
        #         cv2.arrowedLine(line_motion_heatmap, point1, point2_, color, 60, tipLength=0.8)
        #     else:
        #         cv2.line(line_motion_heatmap, point1, point2 , color, 60)
        
        prev_point = [x, y, w, h]
    #line_motion_heatmap = line_motion_heatmap / np.max(line_motion_heatmap)
    line_motion_heatmap = line_motion_heatmap.astype(np.float32)
    return line_motion_heatmap

def get_heatmap(info):
    track, track_id = info

    motion_heatmap = get_motion_heatmap(imgpath, track_id, track['boxes'])

    img_name = save_heatmap_dir + "/%s.jpg" % track_id
    #print(img_name)
    cv2.imwrite(img_name, motion_heatmap )
    print(f"motion: {img_name} save done")

all_tracks_ids = list(all_tracks.keys())

files = []
for track_id in all_tracks:
    files.append((all_tracks[track_id], track_id))




with multiprocessing.Pool(n_worker) as pool:
    for imgs in tqdm(pool.imap_unordered(get_heatmap, files)):
        pass