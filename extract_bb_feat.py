import json
import math
import os
import sys
from datetime import datetime
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from utils_ import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer
import pickle
from collections import OrderedDict
from utils_ import MgvSaveHelper
from PIL import Image
from models.siamese_baseline import SiameseLocalandMotionModelBIG

ossSaver = MgvSaveHelper()


def main():
    config_path = 'configs/single_baseline_aug1_plus.yaml'
    with open('scripts/data/test_track_bboxes.json', 'r') as fb:
        test_bboxes = json.load(fb)
    with open('data/AIC23_Track2_NL_Retrieval/data/test_nlpaug.json', 'r') as fb:
        test_tracks = json.load(fb)
    with open('scripts/data/test_query_cars.json', 'r') as fb:
        test_query_cars = json.load(fb)

    cfg = get_default_config()
    cfg.merge_from_file(config_path)
    ossSaver.set_stauts(save_oss=True, oss_path=cfg.DATA.OSS_PATH)
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
    checkpoint = ossSaver.load_pth(cfg.TEST.RESTORE_FROM)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.cuda()
    torch.backends.cudnn.benchmark = True

    test_data = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test)
    testloader = DataLoader(dataset=test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8)

    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

    model.eval()

    query_car_embeds = dict()
    with torch.no_grad():
        for query_id, texts in tqdm(test_query_cars.items()):
            query_car_embeds[query_id] = []
            car_tokens = tokenizer.batch_encode_plus(texts, padding='longest', return_tensors='pt')
            car_embeds = model.encode_text(car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda())[0]
            query_car_embeds[query_id] = car_embeds.data.cpu().numpy()

    track_car_embeds = dict()
    with torch.no_grad():
        for track_id in tqdm(test_bboxes.keys()):
            track = test_tracks[track_id]
            bboxes = test_bboxes[track_id]
            crops = []
            for frame in bboxes.keys():
                list_of_boxes = bboxes[frame]
                frame_path = os.path.join('data/AIC23_Track2_NL_Retrieval/data', frame)
                image = Image.open(frame_path)
                if len(list_of_boxes) == 1:
                    for box in list_of_boxes:
                        crop = image.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                        crop = transform_test(crop)
                        crops.append(crop)
                else:
                    for box in list_of_boxes[1:]:
                        crop = image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
                        crop = transform_test(crop)
                        crops.append(crop)

            crops = torch.stack(crops).cuda()

            motion = torch.rand(crops.shape).cuda()
            vis_embeds = model.encode_images(crops, motion)[0]
            track_car_embeds[track_id] = vis_embeds.data.cpu().numpy()

    with open('data/query_lang_embeds.pkl', 'wb') as fb:
        pickle.dump(query_car_embeds, fb)

    with open('data/track_car_embeds.pkl', 'wb') as fb:
        pickle.dump(track_car_embeds, fb)


if __name__ == '__main__':
    main()