import copy
import json
import os
import pdb

import numpy as np
import pandas as pd
import torch

from metric import *


# detect whether the track stops
def has_stop(boxes):
    nums = len(boxes)
    area = np.zeros(nums)
    center = np.zeros((nums, 2))
    distance = np.zeros(nums - 1)
    for i in range(nums):
        bb = boxes[i]
        area[i] = bb[2] * bb[3]
        center[i, 0] = bb[0] + bb[2] / 2
        center[i, 1] = bb[1] + bb[3] / 2
        if i < nums - 1:
            distance[i] = abs(boxes[i + 1][0] - bb[0]) + abs(boxes[i + 1][1] - bb[1])
    mean_area = np.mean(area)
    start_area = np.mean(area[:3])
    end_area = np.mean(area[-3:])
    if start_area > end_area:
        start_pos = 0
        end_pos = np.where(area < 1.1 * end_area)[0][0]
    else:
        start_pos = np.where(area < 1.1 * start_area)[0][-1]
        end_pos = nums - 1
    for i in range(start_pos, end_pos):
        if distance[i] < 1:
            return 1
    return 0


### ----- 1. load data ------ ###
test_tracks_pth = "./data/test_tracks.json"

save_dir = "./results"
os.makedirs(save_dir,exist_ok=True)

# create the map
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

# load the camerate information of test tracks
track_cam_info = dict()
with open('./data/test_tracks_cam_info.txt', 'r') as f:
    all_track_cams = f.readlines()
    for cur_track in all_track_cams:
        cur_track = cur_track.strip()
        track_uuid = cur_track.split(' ')[0]
        caminfo = cur_track.split(' ')[1]
        track_cam_info[track_uuid] = caminfo

# load the generated nls by our speaker model for each tracks 
speaker_nl_dict = {}
with open('data/speaker_track2_test_nl_v2.txt', 'r') as f:
    for line in f.readlines():
        track_uuid, nl = line.strip().split(': ')
        nl = nl.replace(' .', '.')
        speaker_nl_dict[track_uuid] = nl


# load the test queries and test tracks
test_queries_nl = json.load(open('./data/test_queries.json'))
test_trackes_bboxes = json.load(open('./data/test_tracks.json'))


# load the info of the sim_mat 
test_tracks = json.load(open(test_tracks_pth))
test_track_ids = list(test_tracks.keys())

# pay attention to the corresonding of the track id and the query id in test
test_query_ids = list()
for track_id in test_track_ids:
    test_query_ids.append(track_to_query[track_id])

### ----- 2. rescore the similarity matrix by adding constraints of some particular scenes ------ ###
cam_sbend = ['c017']
cam_parking = ['c027', 'c012']
cam_busy_intersection = ['c002', 'c003', 'c004', 'c005', 'c026', 'c030', 'c033', 'c034', 'c036']


# scene confusion matrix 

conf_sbend_mat = np.zeros((184,184))
conf_parking_mat = np.zeros((184,184))
conf_busy_mat = np.zeros((184,184))
conf_stop_mat = np.zeros((184,184))
conf_turn_mat = np.zeros((184,184))
conf_left_simmat = np.zeros((184,184))
conf_right_simmat = np.zeros((184,184))
conf_clip_feats_mat = torch.load('data/clip_feats_mat.pth').cpu().detach().numpy()

for i,query_id in enumerate(test_query_ids):
    cur_nl = test_queries_nl[query_id]['nl']
    scene = 'None'
    query_stop_flag = 0
    query_turn_flag = 0
    query_left_flag = 0
    query_right_flag = 0
    # scene strategy
    for each_nl in cur_nl:
        if 'S-bend' in each_nl:
            scene = 'S-bend'
            break
        if 'parking lot' in each_nl:
            scene = 'parking'
            break
        if 'busy intersection' in each_nl:
            scene = 'busy'
            break
    # stop and turn strategy
    for nl in cur_nl:
        if not query_stop_flag and (nl.find("stops at") != -1 or nl.find("stopped at") != -1):
            query_stop_flag = 1

        if not query_turn_flag and nl.find("turn") != -1:
            query_turn_flag = 1

        if not query_left_flag and nl.find("left") != -1:
            query_left_flag = 1

        if not query_right_flag and nl.find("right") != -1:
            query_right_flag = 1



    for j,track_id in enumerate(test_track_ids):
        
        # scene strategy
        cur_cam = track_cam_info[track_id]
        if cur_cam in cam_sbend and scene == 'S-bend':
            conf_sbend_mat[i, j] = 1
        if cur_cam in cam_parking and scene == 'parking':
            conf_parking_mat[i, j] = 1
        if cur_cam in cam_busy_intersection and scene == 'busy':
           conf_busy_mat[i, j] = 1

        # stop and turn strategy
        bboxes = test_trackes_bboxes[track_id]["boxes"]
        track_stop_flag = has_stop(bboxes)
        track_left_flag = 0
        track_right_flag = 0
        if speaker_nl_dict[track_id].find("turn") != -1:
            track_turn_flag = 1
        else:
            track_turn_flag = 0

        if speaker_nl_dict[track_uuid].find("left") != -1:
            track_left_flag = 1
        if speaker_nl_dict[track_uuid].find("right") != -1:
            track_right_flag = 1

        # simmat
        if query_stop_flag and track_stop_flag:
            conf_stop_mat[i, j] = 1
        if query_turn_flag and track_turn_flag:
            conf_turn_mat[i, j] = 1
        if query_left_flag and track_left_flag:
            # print('in left')
            conf_left_simmat[i, j] = 1
        if query_right_flag and track_right_flag:
            print('in right')
            conf_right_simmat[i, j] = 1

# pdb.set_trace()
print('Post process by add scene condition')

# load the processed similarity matrix by post-process part1
rescore_sim_mat_2 = torch.load('../post-process-part1/sim_mat/rescore_sim_mat_soup_pseudo.pth')

### emperically set the weight
best_bw = 0.2
best_sw = 0.2
best_pw = 0.4
best_stw = 0.15
best_tw = 0.03
best_lll = 0.0
best_rrr = 0.0
best_clip = 0.2

sim_mat = rescore_sim_mat_2 + \
            best_bw * conf_busy_mat + \
            best_sw * conf_sbend_mat + \
            best_pw * conf_parking_mat + \
            best_stw * conf_stop_mat + \
            best_tw * conf_turn_mat  + \
            best_lll * conf_left_simmat + \
            best_rrr * conf_right_simmat + \
            best_clip * conf_clip_feats_mat



torch.save(sim_mat, './sim_mat/rescore_sim_mat_soup.pth')

### get the final submission ###
submit_dict = dict()

rank_res = (-sim_mat).argsort()  # query_ids x track_ids
rank_res = rank_res.astype(int)
for i,query_id in enumerate(test_query_ids):
    submit_dict[query_id] = list(np.array(test_track_ids)[rank_res[i,:]])

with open(os.path.join(save_dir,"submit_11.json"),'w') as f:
    json.dump(submit_dict,f)

