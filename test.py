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
import os.path as osp
import refile

from config import get_default_config
from models import build_model
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer
import pickle
from collections import OrderedDict
from main import prepare_start
from utils import get_mrr, MgvSaveHelper
import IPython


def inference_vis_and_lang(config_name, args, enforced=False):
    cfg = get_default_config()
    path = 'configs/' + config_name + '.yaml'
    cfg.merge_from_file(path)

    checkpoint_name = cfg.TEST.RESTORE_FROM.split('/')[-1].split('.')[0]
    save_dir = 'extracted_feats'

    feat_pth_path = save_dir + '/img_lang_feat_%s.pth' % checkpoint_name

    if args.ossSaver.check_s3_path(feat_pth_path):
        if not enforced and refile.s3_isfile(feat_pth_path):
            return feat_pth_path
    else:
        if not enforced and osp.isfile(feat_pth_path):
            return feat_pth_path

    print(f"====> Generating {feat_pth_path}")

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_data = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test)
    testloader = DataLoader(dataset=test_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, )

    args.resume = True
    args.use_cuda = True
    cfg.MODEL.NUM_CLASS = 2155

    model = build_model(cfg, args)

    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    else:
        assert False

    model.eval()

    index = cfg.MODEL.MAIN_FEAT_IDX

    all_lang_embeds = dict()
    with open(cfg.TEST.QUERY_JSON_PATH) as f:
        print(f"====> Query {cfg.TEST.QUERY_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for text_id in tqdm(queries):
            text = queries[text_id]['nl'][:-1]
            car_text = queries[text_id]['nl'][-1:]

            # same dual Text
            if cfg.MODEL.SAME_TEXT:
                car_text = text

            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            if 'dual-text' in cfg.MODEL.NAME:
                car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                            car_tokens['input_ids'].cuda(),
                                                            car_tokens['attention_mask'].cuda())
            else:
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
            lang_embeds = lang_embeds_list[index]
            all_lang_embeds[text_id] = lang_embeds.data.cpu().numpy()

    all_visual_embeds = dict()
    out = dict()
    with torch.no_grad():
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(testloader)):
            vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda())
            vis_embed = vis_embed_list[index]
            for i in range(len(track_id)):
                if track_id[i] not in out:
                    out[track_id[i]] = dict()
                out[track_id[i]][frames_id[i].item()] = vis_embed[i, :]
        for track_id, img_feat in out.items():
            tmp = []
            for fid in img_feat:
                tmp.append(img_feat[fid])
            tmp = torch.stack(tmp)
            tmp = torch.mean(tmp, 0)
            all_visual_embeds[track_id] = tmp.data.cpu().numpy()



    feats = (all_visual_embeds, all_lang_embeds)

    args.ossSaver.save_pth(feat_pth_path, feats)

    return feat_pth_path


def main():
    args, cfg = prepare_start()

    config_dict = {
        "single_baseline_aug1_plus": 1.,
    }

    config_file_list = list(config_dict.keys())
    merge_weights = list(config_dict.values())

    for config_name in config_file_list:
        vis_pkl, lang_pkl = inference_vis_and_lang(config_name, args, enforced=False)


if __name__ == '__main__':
    main()
