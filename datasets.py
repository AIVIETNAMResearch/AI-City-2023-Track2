import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from utils_ import get_logger
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    #print(intervals)
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        
    #print(ranges)
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= False, motion_transform=None, 
                 years=2022, use_multi_frames=False, frames_concat=False, text_aug=None):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.name = json_path
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.motion_transform = motion_transform if motion_transform else transform
        self.bk_dic = {}
        self._logger = get_logger()
        self.years = years
        self.use_multi_frames = use_multi_frames
        self.frames_concat = frames_concat
        if self.data_cfg.USE_HEATMAP:
            self.heat_dic = {}

        self.text_aug = text_aug

        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False]*len(self.list_of_uuids)
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        # print(len(self.all_indexs))
        print(f"====> {json_path} data load, ids: {len(self.all_indexs)}")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
            nl_view_idx = int(random.uniform(0, len(track["nl_other_views"])))
        else:
            nl_idx = 2
            frame_idx = 0
            nl_view_idx = 0
        text = track["nl"][nl_idx]
        if self.data_cfg.USE_MULTI_QUERIES:
            text = track["nl"][:-1]

        car_text = track["nl"][-1]  # -1 = 3
        if len(track["nl_other_views"]) == 0:
            # empty other view, using others
            rand_idx = int(random.uniform(0, len(self.list_of_tracks)))
            while rand_idx == tmp_index:
                rand_idx = int(random.uniform(0, len(self.list_of_tracks)))
            other_track = self.list_of_tracks[rand_idx]
            view_text = other_track["nl"][nl_idx]
        else:
            view_text = track["nl_other_views"][nl_view_idx]
        
        if flag:
            if self.data_cfg.USE_MULTI_QUERIES:
                text = [t.replace("left","888888").replace("right","left").replace("888888","right") for t in text]
            else:
                text = text.replace("left","888888").replace("right","left").replace("888888","right")
        
        if self.use_multi_frames:
            frames_idxes = sample_frames(self.data_cfg.NUM_FRAMES, len(track["frames"]), sample='uniform')

            crops = []
            for frame_idx_ in frames_idxes:
                frame = default_loader(os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx_]))
            
                box = track["boxes"][frame_idx_]
                if self.crop_area == 1.6666667:
                    box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
                else:
                    box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
                
                crop = frame.crop(box)
                if self.transform is not None:
                    crop = self.transform(crop)
                crops.append(crop)


            if self.data_cfg.USE_MOTION:
                if self.list_of_uuids[tmp_index] in self.bk_dic:
                    bk = self.bk_dic[self.list_of_uuids[tmp_index]]
                else:
                    motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
                    bk = default_loader(motion_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                    self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                    bk = self.motion_transform(bk)
                    
                if flag:
                    crops = [torch.flip(crop,[1]) for crop in crops]
                    bk = torch.flip(bk,[1])
                
                if self.data_cfg.USE_HEATMAP:
                    if self.list_of_uuids[tmp_index] in self.heat_dic:
                        heatmap = self.heat_dic[self.list_of_uuids[tmp_index]]
                    else:
                        heatmap_path = self.data_cfg.HEATMAP_PATH
                        heatmap = default_loader(heatmap_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                        self.heat_dic[self.list_of_uuids[tmp_index]] = heatmap
                        heatmap = self.motion_transform(heatmap)
                    #print(heatmap.shape)

                    bk = torch.add(bk, heatmap)
                    print(bk.shape)

                tmp_crops = torch.zeros((self.data_cfg.NUM_FRAMES, 3, self.data_cfg.SIZE, self.data_cfg.SIZE))
                crops = torch.stack(crops)
                tmp_crops[:crops.shape[0], :, :, :] = crops
                crops = tmp_crops
                # return crop,text,bk,tmp_index

                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])

                return {
                    "crop_data": crops,
                    "text": text,
                    "car_text": car_text,
                    "view_text": view_text,
                    "bk_data": bk,
                    "tmp_index": tmp_index,
                    "camera_id": 0,
                }
            
        if self.frames_concat:
            frames_idxes = sample_frames(4, len(track["frames"]), sample='uniform')
            concat_crops = torch.zeros((3, self.data_cfg.SIZE*2, self.data_cfg.SIZE*2))
            for idx, frame_idx in enumerate(frames_idxes):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
                frame = default_loader(frame_path)
                box = track["boxes"][frame_idx]
                if self.crop_area == 1.6666667:
                    box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
                else:
                    box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
                
                crop = frame.crop(box)
                if self.transform is not None:
                    crop = self.transform(crop)
                x_indx = idx % 2
                y_indx = idx // 2
                size = self.data_cfg.SIZE
                concat_crops[:, x_indx * size: (x_indx +1) * size, y_indx * size : (y_indx+1) * size] = crop


            resize = torchvision.transforms.Resize(size=(self.data_cfg.SIZE, self.data_cfg.SIZE))
            concat_crops = resize(concat_crops)
            crop = concat_crops

        else:
            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
            
            frame = default_loader(frame_path)
            box = track["boxes"][frame_idx]
            if self.crop_area == 1.6666667:
                box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
            else:
                box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        
            crop = frame.crop(box)
            if self.transform is not None:
                crop = self.transform(crop)

        if self.data_cfg.USE_MOTION:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
                bk = default_loader(motion_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.motion_transform(bk)
                
            if flag:
                crop = torch.flip(crop,[1])
                bk = torch.flip(bk,[1])

            if self.data_cfg.USE_HEATMAP:
                if self.list_of_uuids[tmp_index] in self.heat_dic:
                    heatmap = self.heat_dic[self.list_of_uuids[tmp_index]]
                else:
                    heatmap_path = self.data_cfg.HEATMAP_PATH
                    heatmap = default_loader(heatmap_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                    self.heat_dic[self.list_of_uuids[tmp_index]] = heatmap
                    heatmap = self.motion_transform(heatmap)

                bk = torch.add(bk, heatmap)

            # return crop,text,bk,tmp_index
            if self.data_cfg.USE_CLIP_FEATS:
                clip_feats = torch.load(self.data_cfg.CLIP_PATH+"/%s.pth"%self.list_of_uuids[tmp_index])
                clip_feats_text = torch.Tensor(clip_feats['text'].detach().cpu().numpy())
                clip_feats_vis = torch.Tensor(clip_feats['vis'].detach().cpu().numpy())
                return {
                    "crop_data": crop,
                    "text": text,
                    "car_text": car_text,
                    "view_text": view_text,
                    "bk_data": bk,
                    "tmp_index": tmp_index,
                    "camera_id": 0,
                    "clip_feats_text": clip_feats_text,
                    "clip_feats_vis": clip_feats_vis
                }

            return {
                "crop_data": crop,
                "text": text,
                "car_text": car_text,
                "view_text": view_text,
                "bk_data": bk,
                "tmp_index": tmp_index,
                "camera_id": 0,
            }
        if flag:
            crop = torch.flip(crop,[1])
        # return crop,text,tmp_index
        return {
            "crop_data": crop,
            "text": text,
            "car_text": 0,
            "tmp_index": tmp_index,
            "camera_id": 0,
        }


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg, transform=None, years=2022, val=False, use_multi_frames=False, frames_concat=False):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        self.years = years
        self.use_multi_frames = use_multi_frames
        self.frames_concat = frames_concat

        if val:
            self.name = data_cfg.EVAL_JSON_PATH
            with open(self.data_cfg.EVAL_JSON_PATH) as f:
                tracks = json.load(f)
        else:
            self.name = data_cfg.TEST_TRACKS_JSON_PATH
            with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
                tracks = json.load(f)

        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()


        for track_id_index,track in enumerate(self.list_of_tracks):
            if use_multi_frames or frames_concat:
                num_frames = 4 if frames_concat else self.data_cfg.NUM_FRAMES
                frame_indexes = sample_frames(num_frames, len(track['frames']), sample='uniform')
                frame_paths = [os.path.join(self.data_cfg.CITYFLOW_PATH, track['frames'][idx]) for idx in frame_indexes]
                boxes = [track['boxes'][idx] for idx in frame_indexes]
                crop = {'frames': frame_paths, 'frames_id': frame_indexes[0], "track_id": self.list_of_uuids[track_id_index], "boxes": boxes}
                self.list_of_crops.append(crop)

            else:
                for frame_idx, frame in enumerate(track["frames"]):
                    frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                    box = track["boxes"][frame_idx]
                    crop = {"frames": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "boxes": box}
                    self.list_of_crops.append(crop)
        self._logger = get_logger()
        print(f"====> {self.data_cfg.TEST_TRACKS_JSON_PATH} data load, ids: {len(self.list_of_crops)}")

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        
        if self.use_multi_frames or self.frames_concat:
            frames_path = track['frames']

            crops = []
            for idx, frame_path in enumerate(frames_path):
                frame = default_loader(frame_path)
                    
                box = track["boxes"][idx]
                if self.crop_area == 1.6666667:
                    box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
                else:
                    box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
                
                crop = frame.crop(box)
                if self.transform is not None:
                    crop = self.transform(crop)

                crops.append(crop)

            if self.frames_concat:
                concat_crops = torch.zeros(3, self.data_cfg.SIZE * 2, self.data_cfg.SIZE * 2)
                for idx_, crop in enumerate(crops):
                    x_indx = idx_ % 2
                    y_indx = idx_ // 2
                    size = self.data_cfg.SIZE
                    concat_crops[:, x_indx * size: (x_indx +1) * size, y_indx * size: (y_indx+1) * size] = crop
                crops = concat_crops
                resize = torchvision.transforms.Resize(size=(self.data_cfg.SIZE, self.data_cfg.SIZE))
                concat_crops = resize(concat_crops)
                crop = concat_crops
                
            else:
                
                tmp_crops = torch.zeros((self.data_cfg.NUM_FRAMES, 3, self.data_cfg.SIZE, self.data_cfg.SIZE))
                crops = torch.stack(crops)
                tmp_crops[:crops.shape[0], :, :, :] = crops
                #print(tmp_crops.shape)
                crops = tmp_crops

            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][0])

            #print("motion: ", self.data_cfg.USE_MOTION, "frames: ", self.use_multi_frames, "concat: ", self.data_cfg.FRAMES_CONCAT)
            if self.data_cfg.USE_MOTION:
                motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
                bk = default_loader(motion_path+"/%s.jpg"%track["track_id"])
                bk = self.transform(bk)
                if self.data_cfg.USE_HEATMAP:
                    
                    heatmap_path = self.data_cfg.HEATMAP_PATH
                    heatmap = default_loader(heatmap_path+"/%s.jpg"%track["track_id"])
                    heatmap = self.transform(heatmap)

                    bk = torch.add(bk, heatmap)
                    
                if self.data_cfg.USE_CLIP_FEATS:
                    clip_feats = torch.load(self.data_cfg.CLIP_PATH+"/%s.pth"%self.list_of_uuids[track['track_id']])
                    clip_feats_text = clip_feats['text']
                    clip_feats_vis = clip_feats['vis']
                    return crops,bk,track["track_id"],track["frames_id"], clip_feats_text, clip_feats_vis
                return crops,bk,track["track_id"],track["frames_id"]

            return crops,track["track_id"],track["frames_id"]

        frame_path = track["frames"]

        frame = default_loader(frame_path)
        box = track["boxes"]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
            bk = default_loader(motion_path+"/%s.jpg"%track["track_id"])
            bk = self.transform(bk)

            if self.data_cfg.USE_HEATMAP:
                
                heatmap_path = self.data_cfg.HEATMAP_PATH
                heatmap = default_loader(heatmap_path+"/%s.jpg"%track["track_id"])
                heatmap = self.transform(heatmap)

                bk = torch.add(bk, heatmap)

            if self.data_cfg.USE_CLIP_FEATS:
                clip_feats = torch.load(self.data_cfg.CLIP_PATH+"/%s.pth"%track['track_id'])
                clip_feats_vis = clip_feats['vis']
                return crop,bk,track["track_id"],track["frames_id"], clip_feats_vis
            
            return crop,bk,track["track_id"],track["frames_id"]
        return crop,track["track_id"],track["frames_id"]


class CityFlowNLDatasetVal(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, Random=True, motion_transform=None, years=2022):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.name = json_path
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.motion_transform = motion_transform if motion_transform else transform
        self.bk_dic = {}
        self._logger = get_logger()
        self.years = years

        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False] * len(self.list_of_uuids)
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        # print(len(self.all_indexs))
        print(f"====> {json_path} data load, ids: {len(self.all_indexs)}")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):

        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        uuid = self.list_of_uuids[tmp_index]
        if self.random:
            nl_idx = int(random.uniform(0, 3))
            nl_view_idx = int(random.uniform(0, len(track["nl_other_views"])))
        else:
            nl_idx = 2
            frame_idx = 0
            nl_view_idx = 0
        text = track["nl"][nl_idx]
        car_text = track["nl"][-1]  # -1 = 3
        view_text = track["nl_other_views"][nl_view_idx]
        if flag:
            text = text.replace("left", "888888").replace("right", "left").replace("888888", "right")
        
        if self.use_multi_frames:
            frames_idxes = sample_frames(self.data_cfg.NUM_FRAMES, len(track["frames"]), sample='uniform')
            crops = []
            
            for frame_idx in enumerate(frames_idxes):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
                frame = default_loader(frame_path)
                box = track["boxes"][frame_idx]
                if self.crop_area == 1.6666667:
                    box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
                else:
                    box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
                
                crop = frame.crop(box)
                if self.transform is not None:
                    crop = self.transform(crop)
                crops.append(crop)


            if self.data_cfg.USE_MOTION:
                if self.list_of_uuids[tmp_index] in self.bk_dic:
                    bk = self.bk_dic[self.list_of_uuids[tmp_index]]
                else:
                    motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
                    bk = default_loader(motion_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                    self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                    bk = self.motion_transform(bk)
                    
                if flag:
                    crops = [torch.flip(crop,[1]) for crop in crops]
                    bk = torch.flip(bk,[1])

                if self.data_cfg.USE_HEATMAP:
                    concat_bk = torch.zeros((3, self.data_cfg.SIZE * 2, self.data_cfg.SIZE))
                    
                    heatmap_path = self.data_cfg.HEATMAP_PATH
                    heatmap = default_loader(heatmap_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                    heatmap = self.motion_transform(heatmap)

                    concat_bk[:, :self.data_cfg.SIZE, :self.data_cfg.SIZE] = bk
                    concat_bk[:, self.data_cfg.SIZE: self.data_cfg.SIZE*2, : self.data_cfg.SIZE] = heatmap

                    bk = concat_bk
                
                crops = torch.stack(crops)
                # return crop,text,bk,tmp_index

                return {
                    "crop_data": crops,
                    "text": text,
                    "car_text": car_text,
                    "view_text": view_text,
                    "bk_data": bk,
                    "tmp_index": tmp_index,
                    "camera_id": 0,
                    "uuid": uuid,
                }

        if self.frames_concat:
            frames_idxes = sample_frames(4, len(track["frames"]), sample='uniform')
            concat_crops = torch.zeros((self.data_cfg.SIZE*2, self.data_cfg.SIZE*2))
            for idx in frames_idxes:
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][idx])
                frame = default_loader(frame_path)
                box = track["boxes"][frame_idx]
                if self.crop_area == 1.6666667:
                    box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
                else:
                    box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
                
                crop = frame.crop(box)
                if self.transform is not None:
                    crop = self.transform(crop)
                x_indx = idx % 2
                y_indx = idx // 2
                size = self.data_cfg.SIZE
                concat_crops[x_indx * size: (x_indx +1) * size, y_indx * size, (y_indx+1) * size] = crop    
            
            crop = concat_crops
        else:

            frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
            frame = default_loader(frame_path)
            box = track["boxes"][frame_idx]
            if self.crop_area == 1.6666667:
                box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                    int(box[1] + 4 * box[3] / 3.))
            else:
                box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                    int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

            crop = frame.crop(box)
            if self.transform is not None:
                crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                motion_path = self.data_cfg.MOTION_PATH if self.years == 2022 else self.data_cfg.MOTION_PATH_2021
                bk = default_loader(motion_path + "/%s.jpg" % self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.motion_transform(bk)

            if flag:
                crop = torch.flip(crop, [1])
                bk = torch.flip(bk, [1])
            # return crop, text, bk, tmp_index, uuid
            if self.data_cfg.USE_HEATMAP:
                concat_bk = torch.zeros((3, self.data_cfg.SIZE * 2, self.data_cfg.SIZE))
                
                heatmap_path = self.data_cfg.HEATMAP_PATH
                heatmap = default_loader(heatmap_path+"/%s.jpg"%self.list_of_uuids[tmp_index])
                heatmap = self.motion_transform(heatmap)

                concat_bk[:, :self.data_cfg.SIZE, :self.data_cfg.SIZE] = bk
                concat_bk[:, self.data_cfg.SIZE: self.data_cfg.SIZE*2, : self.data_cfg.SIZE] = heatmap

                bk = concat_bk
            return {
                "crop_data": crop,
                "text": text,
                "car_text": car_text,
                "view_text": view_text,
                "bk_data": bk,
                "tmp_index": tmp_index,
                "camera_id": 0,
                "uuid": uuid,
            }
        if flag:
            crop = torch.flip(crop, [1])
        # return crop, text, tmp_index, uuid
        return {
            "crop_data": crop,
            "text": text,
            "car_text": 0,
            "view_text": 0,
            "tmp_index": tmp_index,
            "camera_id": 0,
            "uuid": uuid,
        }
        