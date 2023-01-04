import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from utils import get_logger

def default_loader(path):
    return Image.open(path).convert('RGB')


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True, motion_transform=None, years=2022):
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
            text = text.replace("left","888888").replace("right","left").replace("888888","right")
        
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
            # return crop,text,bk,tmp_index
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
    def __init__(self, data_cfg, transform=None, years=2022, val=False):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        self.years = years
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
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()
        print(f"====> {self.data_cfg.TEST_TRACKS_JSON_PATH} data load, ids: {len(self.list_of_crops)}")

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        frame = default_loader(frame_path)
        box = track["box"]
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