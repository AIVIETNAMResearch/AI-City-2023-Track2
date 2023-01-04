import json
import os 
import pickle
import numpy as np
import torch
from numpy import linalg as LA
import torch
import torch.nn.functional as F
import os.path as osp

import IPython

from main import prepare_start
from test import inference_vis_and_lang


# with open("data2022/test-queries_nlpaug.json") as f:
with open("data2022/test-queries.json") as f:
    queries = json.load(f)
with open("data2022/test-tracks.json") as f:
    tracks = json.load(f)
query_ids = list(queries.keys())
tacks_ids = list(tracks.keys())
print(len(tacks_ids), len(query_ids))


def get_lang_v(texts):
    location = 0
    direction = 0

    num_left = 0
    num_right = 0
    for text in texts:
        if 'turn' in text:
            if 'left' in text:
                num_left += 1
            if 'right' in text:
                num_right += 1
        if 'intersection' in text:
            location = 1

    if num_left > num_right:
        direction = 1
    if num_left < num_right:
        direction = 2

    loc_map = [[1, 0], [0, 1]]
    dir_map = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return loc_map[location], dir_map[direction]


def get_mean_feats(img_feat, tacks_ids):
    mean_gallery = []
    for k in tacks_ids:
        tmp = []
        for fid in img_feat[k]:
            tmp.append(img_feat[k][fid])
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp,0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery


def feature_mean_score_test(args, config_list, merge_weights, enforced=False, spatial=False):
    img_feats = []
    nlp_feats = []
    for i, cur_config in enumerate(config_list):
        feat_pth_path = inference_vis_and_lang(cur_config, args, enforced)
        vis_feats, lang_ems = args.ossSaver.load_pth(feat_pth_path)
        vis_feats = [v for k, v in vis_feats.items()]
        img_feats.append(vis_feats)
        nlp_feats.append(lang_ems)

    if spatial:
        # # Location similarity (long-distance spatial relationship)
        loc_dict = {'c005': 1, 'c014': 0, 'c029': 0, 'c003': 1, 'c017': 0, 'c035': 0, 'c013': 0, 'c027': 0, 'c038': 0,
                    'c016': 0, 'c002': 1, 'c021': 0, 'c019': 0, 'c030': 0, 'c020': 0, 'c026': 0, 'c010': 0, 'c004': 1,
                    'c033': 0, 'c012': 1, 'c001': 1, 'c034': 0, 'c022': 0, 'c036': 0, 'c040': 1, 'c025': 0, 'c032': 0,
                    'c037': 1}
        loc_logits = dict()
        for track_id, track in tracks.items():
            cam = track['frames'][0].split('/')[-3]
            if loc_dict[cam] == 1:
                loc_logits[track_id] = [0, 1]
            else:
                loc_logits[track_id] = [1, 0]

        img_loc_logits = np.array([v for k, v in loc_logits.items()])

        lang_loc_v = dict()
        lang_dir_v = dict()
        for q_id, record in queries.items():
            texts = queries[q_id]["nl"]
            loc_v, dir_v = get_lang_v(texts)
            lang_loc_v[q_id] = loc_v
            lang_dir_v[q_id] = dir_v

        # # Multi-vehicle similarity (short-distance spatial relationship)
        # load feature of vehicle in text
        with open('spatial_feat/query_lang_embeds.pkl', "rb") as fb:
            query_lang_embeds = pickle.load(fb)

        # load feature of bbox in tracks
        with open('spatial_feat/track_car_embeds.pkl', "rb") as fb:
            track_car_embeds = pickle.load(fb)

    results = dict()

    for query in query_ids:
        score = 0.
        for i in range(len(nlp_feats)):
            q = nlp_feats[i][query]
            cur_sim = np.mean(np.matmul(q, np.array(img_feats[i]).T), 0)

            if spatial:
                car_lang_embeds = query_lang_embeds[query]
                if len(car_lang_embeds) >= 2:
                    relation_sim = []
                    other_lang_embed = car_lang_embeds[1]
                    for _, car_vis_embeds in track_car_embeds.items():
                        max_sim_val = 0
                        for car_vis_embed in car_vis_embeds:
                            other_lang_embed = other_lang_embed / np.linalg.norm(other_lang_embed)
                            car_vis_embed = car_vis_embed / np.linalg.norm(car_vis_embed)
                            max_sim_val = max(max_sim_val, np.matmul(other_lang_embed, car_vis_embed.T))
                        relation_sim.append(max_sim_val)
                    relation_sim = np.array(relation_sim)
                    relation_sim = relation_sim / np.max(relation_sim) * np.max(cur_sim) * 0.233333
                    cur_sim = cur_sim + relation_sim

                loc_sim = np.matmul(np.array([lang_loc_v[query]]), img_loc_logits.T)
                loc_sim = loc_sim.T
                loc_sim = np.squeeze(loc_sim)
                cur_sim = cur_sim + loc_sim
            score += merge_weights[i] * cur_sim

        index = np.argsort(score)[::-1]
        results[query] = []
        for i in index:
            results[query].append(tacks_ids[i])

    os.makedirs(args.cfg.DATA.TEST_OUTPUT, exist_ok=True)
    save_path = osp.join(args.cfg.DATA.TEST_OUTPUT, args.save_name)
    with open(save_path, "w") as fs:
        json.dump(results, fs, indent=4)
        print(f"====> save {save_path} done")


def main():
    args, cfg = prepare_start()

    config_dict = {
        'view_triplet_hard': 1,
        'single_baseline_aug1_plus': 1,
        'dual_baseline_aug1': 1,
        'dual_baseline_aug3': 1,
        'circle_loss': 1,
        'single_baseline_aug2': 1,
    }

    config_file_list = list(config_dict.keys())
    merge_weights = list(config_dict.values())

    feat = -1
    args.save_name = f'ensemble_spatial_modeling.json'
    args.cfg.MODEL.MAIN_FEAT_IDX = feat
    # spatial = False
    spatial = True

    # enforced = True
    enforced = False

    feature_mean_score_test(args, config_file_list, merge_weights, enforced=enforced, spatial=spatial)


if __name__ == '__main__':
    main()
