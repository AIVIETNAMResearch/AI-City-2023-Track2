import copy
import json
import os
import pdb

import numpy as np
import pandas as pd
import torch

from metric import *

ori_sim_mat_pth = "./sim_mat/sim_mat.npy"
ori_sim_mat = np.load(ori_sim_mat_pth)

test_tracks_pth = "./data/test_tracks.json"

save_dir = "./results"
os.makedirs(save_dir,exist_ok=True)


# create the track to query map, in our experiments we concatenate the tracks 
# and queries in original order
track_to_query = dict()
query_to_track = dict()

with open("./data/all_track_to_query.txt",'r') as f:
    pairs = f.readlines()
    for pair in pairs:
        pair = pair.strip("\n").strip()
        track_id = pair.split(',')[0]
        query_id = pair.split(',')[1]
        track_to_query[track_id] = query_id
        query_to_track[query_id] = track_id

### 1. load the query labels ###
# load the parsed color label of test query
col_query = pd.read_csv("./data/col_test_one_hot.csv")
col_query = col_query.set_index('query_id')
col_query = col_query['labels']
col_query = col_query.to_dict()
for key,value in col_query.items():
    col_query[key] = eval(value)

# load the parsed type label of test query
veh_query = pd.read_csv("./data/veh_test_one_hot.csv")
veh_query = veh_query.set_index('query_id')
veh_query = veh_query['labels']
veh_query = veh_query.to_dict()
for key,value in veh_query.items():
    veh_query[key] = eval(value)

# load the parsed motion label of test query
motion_query = json.load(open("./data/pure_motion_query.json"))

### 2. load the target track attribtue prediction ###
# load the predicted color of test track
col_track = json.load(open("./data/target_test_color_predict_one_hot.json"))
# load the predicted type of test track
veh_track = json.load(open("./data/target_test_vehicle_predict_one_hot.json"))
# load the predicted motion of test track
motion_track = json.load(open("./data/target_test_direction_predict_one_hot.json"))


# load the info of the simmat 
test_tracks = json.load(open(test_tracks_pth))

test_track_ids = list(test_tracks.keys())
# pay attention to the corresonding of the track id and the query id in test
test_query_ids = list()
for track_id in test_track_ids:
    test_query_ids.append(track_to_query[track_id])


### 3. adjust the sim_mat by adding constraints on the target track's attributes ###
conf_col_mat = np.zeros((184,184))
conf_veh_mat = np.zeros((184,184))
conf_motion_mat = np.zeros((184,184))

for i,query_id in enumerate(test_query_ids):
    for j,track_id in enumerate(test_track_ids):
        # target color constraints
        if query_id in col_query:
            query_col = np.array(col_query[query_id])
            track_col = np.array(col_track[track_id])
            if query_col.argmax() == track_col.argmax():
                conf_col_mat[i,j] = 1
            else:
                conf_col_mat[i,j] = -1

        # target type constraints
        if query_id in veh_query:
            query_veh = np.array(veh_query[query_id])
            track_veh = np.array(veh_track[track_id])
            if query_veh.argmax() == track_veh.argmax():
                conf_veh_mat[i,j] = 1
            elif query_veh.argmax() in [4,5] and track_veh.argmax() in [4,5]:
                conf_veh_mat[i,j] = 1
            else:
                conf_veh_mat[i,j] = -1
        
        # target direction constraints
        # pay attention there maybe some query don't have the vehicle direction attribute
        query_motion = np.array(motion_query[query_id])
        if query_motion.sum() > 1e-5:
            track_motion = np.array(motion_track[track_id])
            if query_motion.argmax() == track_motion.argmax():
                conf_motion_mat[i,j] = 1
            else:
                conf_motion_mat[i,j] = -1


### 4. adjust the sim_mat by adding constraints on the target vehicle related vehicles' attributes ###
# load the pre-computed confusion matrix of the front, back and nearby tracks
confusion_matrix_back_type = np.load("./relation/back_type_mat.npy")
confusion_matrix_back_color = np.load("./relation/back_color_mat.npy")
confusion_matrix_front_type = np.load("./relation/front_type_mat.npy")
confusion_matrix_front_color = np.load("./relation/front_color_mat.npy")

confusion_matrix_near_type = np.load("./relation/near_type_mat.npy")
confusion_matrix_intersection = np.load("./relation/intersection_mat.npy")


### emperically set the weight ###
submit_dict = dict()
best_col = 0.05
best_veh = 0.05
best_motion = 0.1
best_b_col = 0.20 
best_b_veh = 0.20
best_f_col = 0.20
best_f_veh = 0.20
best_inter = 0.1
best_n_veh = 0.1

sim_mat = ori_sim_mat + \
          best_col*conf_col_mat + \
          best_veh*conf_veh_mat+ \
          best_motion*conf_motion_mat+ \
          best_b_col*confusion_matrix_back_color + \
          best_f_col*confusion_matrix_front_color + \
          best_b_veh*confusion_matrix_back_type + \
          best_f_veh*confusion_matrix_front_type + \
          best_inter*confusion_matrix_intersection + \
          best_n_veh*confusion_matrix_near_type

# # save similarity matrix for second stage post-process
torch.save(sim_mat, './sim_mat/rescore_sim_mat_soup_pseudo.pth')


 
 