import json
import math
import os
import sys
import time
from datetime import timedelta, datetime
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
import tabulate
from termcolor import colored
from torch.backends import cudnn
import copy
import numpy as np
import random

from config import get_default_config
from models import build_model
from utils_ import TqdmToLogger, Logger, AverageMeter, accuracy, ProgressMeter
from utils_ import get_mrr, MgvSaveHelper, set_seed
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaTokenizer, RobertaModel
from preprocessing.transforms import build_transforms, build_vanilla_transforms, build_motion_transform, BackTranslateAug
from optimizer.build import build_vanilla_optimizer, FreezeBackbone
from loss import cross_entropy, CircleLoss, TripletLoss, CirclePairLoss, CosFacePairLoss
from rerank import rerank_params_grid_search

import IPython

best_mrr_eval = 0.
best_mrr_eval_by_test = 0.

table = []
max_record = ['Max', 'Max', 0, 0, 0, 0, 0]
header = ['Method', 'Dataset', 'Epoch', 'Loss', 'MRR', 'Acc-1', 'Acc-5']
table.append(header)
print_interval = 20  # csv
save_interval = 200   # checkpoint


def results_record(name, dataset, epoch, losses, mrr, top1_acc, top5_acc, is_test=False):
    # result csv
    record = list()
    # name = args.name
    record.append(name)
    record.append(dataset)
    record.append(epoch)
    record.append(losses)
    record.append(mrr)
    record.append(top1_acc)
    record.append(top5_acc)
    table.append(record)
    print_table = copy.deepcopy(table)
    global max_record
    if is_test and record[-3] > max_record[-3]:
        max_record = copy.deepcopy(record)
        max_record[2] = 'Max_' + str(max_record[2])
    print_table.append(max_record)

    display = tabulate.tabulate(
        print_table,
        tablefmt="pipe",
        headers='firstrow',
        numalign="left",
        floatfmt='.3f')
    print(f"====> results in csv format: \n" + colored(display, "cyan"))


def prepare_start():
    parser = argparse.ArgumentParser(description='AICT5 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', default="configs/single_baseline_aug1.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str, 
                        help='experiments')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/')
    parser.add_argument('--eval_only', '-eval', action='store_true', help='only eval')
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None, 
        nargs=argparse.REMAINDER, 
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    args.cfg = cfg

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    print(f"====> load config from {args.config}")

    ossSaver = MgvSaveHelper()
    ossSaver.set_stauts(save_oss=cfg.DATA.USE_OSS, oss_path=cfg.DATA.OSS_PATH)
    args.ossSaver = ossSaver

    return args, cfg


def evaluate_by_test_all(model, valloader, epoch, cfg, index=-1, args=None, tokenizer=None, optimizer=None, multi_frames=False):
    """ evaluate crop, motion and merge features"""
    global best_mrr_eval_by_test
    print(f"====> Test::::{valloader.dataset.name}")
    evl_start_time = time.monotonic()
    model.eval()

    feat_num = 3
    all_visual_embeds = [dict() for _ in range(feat_num)]
    out = [dict() for _ in range(feat_num)]
    with torch.no_grad():
        if multi_frames:
            for batch_idx, (frames, motion, image, track_id, frames_id) in tqdm(enumerate(valloader)):
                vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda(), frames.cuda())
                for i in range(len(track_id)):
                    for feat_idx in range(feat_num):
                        vis_embed = vis_embed_list[feat_idx]
                        all_visual_embeds[feat_idx][track_id[i]] = vis_embed[i, :]
        else:
            for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader)):
                vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda())
                for i in range(len(track_id)):
                    for feat_idx in range(feat_num):
                        vis_embed = vis_embed_list[feat_idx]
                        if track_id[i] not in out:
                            out[feat_idx][track_id[i]] = dict()
                        out[feat_idx][track_id[i]][frames_id[i].item()] = vis_embed[i, :]
            for track_id in out[-1].keys():
                for feat_idx in range(feat_num):
                    img_feat = out[feat_idx][track_id]
                    tmp = []
                    for fid in img_feat:
                        tmp.append(img_feat[fid])
                    tmp = torch.stack(tmp)
                    tmp = torch.mean(tmp, 0)
                    all_visual_embeds[feat_idx][track_id] = tmp

    all_lang_embeds = [dict() for _ in range(feat_num)]
    with open(cfg.DATA.EVAL_JSON_PATH) as f:
        print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for q_id in tqdm(queries.keys()):
            text = queries[q_id]['nl'][:-1]
            car_text = queries[q_id]['nl'][-1:]

            # same dual Text
            if cfg.MODEL.SAME_TEXT:
                car_text = text

            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            if 'dual-text' in cfg.MODEL.NAME:
                car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                if cfg.DATA.USE_MULTI_QUERIES:
                    lang_embeds_list = model.module.encode_text(torch.unsqueeze(tokens['input_ids'].cuda(), dim=1), 
                                                                torch.unsqueeze(tokens['attention_mask'].cuda(), dim=1),
                                                                car_tokens['input_ids'].cuda(),
                                                                car_tokens['attention_mask'].cuda())
                else:
                    lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), 
                                                                tokens['attention_mask'].cuda(),
                                                                car_tokens['input_ids'].cuda(),
                                                                car_tokens['attention_mask'].cuda())
            else:
                if cfg.DATA.USE_MULTI_QUERIES:
                    lang_embeds_list = model.module.encode_text(torch.unsqueeze(tokens['input_ids'].cuda(), dim=1), 
                                                                torch.unsqueeze(tokens['attention_mask'].cuda(), dim=1))
                else:
                    lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
            for feat_idx in range(feat_num):
                if cfg.DATA.USE_MULTI_QUERIES:
                    lang_embeds = lang_embeds_list[0]
                else:
                    lang_embeds = lang_embeds_list[feat_idx]

                all_lang_embeds[feat_idx][q_id] = lang_embeds

    all_sim = [list() for _ in range(feat_num)]
    with torch.no_grad():
        visual_embeds = [list() for _ in range(feat_num)]
        for q_id in all_visual_embeds[-1].keys():
            for feat_idx in range(feat_num):
                visual_embeds[feat_idx].append(all_visual_embeds[feat_idx][q_id])
        visual_embeds = [torch.stack(embeds) for embeds in visual_embeds]
        for q_id in tqdm(all_visual_embeds[-1].keys()):
            for feat_idx in range(feat_num):
                lang_embeds = all_lang_embeds[feat_idx][q_id]
                cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds[feat_idx].T), 0, keepdim=True)
                all_sim[feat_idx].append(cur_sim)

    all_sim = [torch.cat(sim) for sim in all_sim]

    def compute_and_record(sim, name):
        sim_t_2_i = sim
        sim_i_2_t = sim_t_2_i.t()

        loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
        loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_t_2_i.size(0)).cuda())
        loss = (loss_t_2_i + loss_i_2_t) / 2

        acc1, acc5 = accuracy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda(), topk=(1, 5))
        mrr_ = get_mrr(sim_t_2_i)
        all_mrr = mrr_.item() * 100
        results_record(name, valloader.dataset.name, epoch, loss.item(),
                       all_mrr, acc1[0], acc5[0], is_test=True)
        return all_mrr

    sum_sim = 0.
    for idx, sim in enumerate(all_sim):
        sum_sim += sim
        compute_and_record(sim, args.logs_dir.split('/')[-1] + f'feat_{idx}')
    sum_mrr = compute_and_record(sum_sim, args.logs_dir.split('/')[-1] + 'feat_all')

    if args.eval_only and cfg.TEST.RERANK:
        rerank_mrr, rerank_params, rerank_sim = rerank_params_grid_search(all_lang_embeds[-1], all_visual_embeds[-1])
        print(f"====> grid search rerank, best mrr = {rerank_mrr}, params: k1={rerank_params[0]}, k2={rerank_params[1]}, eps={rerank_params[2]}")

    evl_end_time = time.monotonic()
    print(f'Epoch {epoch} running time: ', timedelta(seconds=evl_end_time - evl_start_time))
    print(f'Logs dir: {args.logs_dir} ')

    if args.eval_only:
        return
    if sum_mrr > best_mrr_eval_by_test:
        # save time
        best_mrr_eval_by_test = sum_mrr
        checkpoint_file = args.logs_dir + "/checkpoint_best_eval_all.pth"
        args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def evaluate_by_test(model, valloader, epoch, cfg, index=-1, args=None, tokenizer=None, optimizer=None, multi_frames=False):
    """ evaluate merge features"""
    global best_mrr_eval_by_test
    print(f"====> Test::::{valloader.dataset.name}")
    evl_start_time = time.monotonic()
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Clip Loss', ':.4e')
    mrr = AverageMeter('MRR', ':6.4f')
    top1_acc = AverageMeter('Acc@1', ':6.4f')
    top5_acc = AverageMeter('Acc@5', ':6.4f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses, mrr, top1_acc, top5_acc],
        prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()

    all_visual_embeds = dict()
    out = dict()
    with torch.no_grad():
        if multi_frames:
            for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader)):
                vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda())
                vis_embed = vis_embed_list[index]
                for i in range(len(track_id)):
                    all_visual_embeds[track_id[i]] = vis_embed[i, :]
        elif cfg.DATA.USE_CLIP_FEATS:
            for batch_idx, (image, motion, track_id, frames_id, clip_feats_text, clip_feats_vis) in tqdm(enumerate(valloader)):
                vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda(), 
                                                            clip_feats_vis=clip_feats_vis.cuda())
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
                all_visual_embeds[track_id] = tmp
        else:
            for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader)):
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
                all_visual_embeds[track_id] = tmp


    all_lang_embeds = dict()
    with open(cfg.DATA.EVAL_JSON_PATH) as f:
        print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for q_id in tqdm(all_visual_embeds.keys()):
            text = queries[q_id]['nl'][:-1]
            car_text = queries[q_id]['nl'][-1:]
            if cfg.DATA.USE_CLIP_FEATS:
                clip_feats = torch.load(cfg.DATA.CLIP_PATH+"/%s.pth"%q_id)
                clip_feats_text = clip_feats['text']

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
                if cfg.DATA.USE_CLIP_FEATS:

                    lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                                clip_feats_text=clip_feats_text.cuda())
                elif cfg.DATA.USE_MULTI_QUERIES:
                    lang_embeds_list = model.module.encode_text(torch.unsqueeze(tokens['input_ids'].cuda(), dim=1), 
                                                                torch.unsqueeze(tokens['attention_mask'].cuda(), dim=1))
                
                else:
                    lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())

            lang_embeds = lang_embeds_list[index]
            all_lang_embeds[q_id] = lang_embeds

    all_sim = []
    with torch.no_grad():
        visual_embeds = []
        for q_id in all_visual_embeds.keys():
            visual_embeds.append(all_visual_embeds[q_id])
        visual_embeds = torch.stack(visual_embeds)
        for q_id in tqdm(all_visual_embeds.keys()):
            lang_embeds = all_lang_embeds[q_id]
            cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds.T), 0, keepdim=True)
            all_sim.append(cur_sim)

    if args.eval_only and cfg.TEST.RERANK:
        rerank_mrr, rerank_params, rerank_sim = rerank_params_grid_search(all_lang_embeds, all_visual_embeds)
        print(f"====> grid search rerank, best mrr = {rerank_mrr}, params: k1={rerank_params[0]}, k2={rerank_params[1]}, eps={rerank_params[2]}")

    all_sim = torch.cat(all_sim)
    sim_t_2_i = all_sim
    sim_i_2_t = sim_t_2_i.t()

    loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
    loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_t_2_i.size(0)).cuda())
    loss = (loss_t_2_i + loss_i_2_t) / 2

    acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(1, 5))
    mrr_ = get_mrr(all_sim)
    all_mrr = mrr_.item() * 100

    losses.update(loss.item(), image.size(0))
    mrr.update(all_mrr, image.size(0))
    top1_acc.update(acc1[0], image.size(0))
    top5_acc.update(acc5[0], image.size(0))
    batch_time.update(time.time() - end)

    progress.display(batch_idx)

    evl_end_time = time.monotonic()
    print(f'Epoch {epoch} running time: ', timedelta(seconds=evl_end_time - evl_start_time))
    print(f'Logs dir: {args.logs_dir} ')

    results_record(args.logs_dir.split('/')[-1], valloader.dataset.name, epoch, loss.item(), all_mrr, acc1[0], acc5[0], is_test=True)
    if args.eval_only:
        return
    if all_mrr > best_mrr_eval_by_test:
        # save time
        best_mrr_eval_by_test = all_mrr
        checkpoint_file = args.logs_dir + "/checkpoint_best_eval.pth"
        args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def main():
    args, cfg = prepare_start()
    if cfg.MODEL.METRIC.LOSS == '':
        set_seed(cfg.TRAIN.SEED, cfg.TRAIN.DETERMINISTIC)
    os.makedirs(args.logs_dir, exist_ok=True)
    # print(cfg)

    if not cfg.DATA.CROP_AUG:
        # CLV(1st) transforms
        transform_train = build_vanilla_transforms(cfg, is_train=True)
        transform_test = build_vanilla_transforms(cfg, is_train=False)
        motion_transform = build_vanilla_transforms(cfg, is_train=True)
    else:
        # CLV(1st) and DUN(2st)
        transform_train = build_transforms(cfg, is_train=True)
        transform_test = build_transforms(cfg, is_train=False)
        motion_transform = build_motion_transform(cfg, vanilla=True)

    text_aug = None
    if cfg.DATA.TEXT_AUG:
        text_aug = BackTranslateAug()

    print("Using multi frames: ", cfg.DATA.MULTI_FRAMES)
    print("Using frames concat: ", cfg.DATA.FRAMES_CONCAT)
    print("Using heatmap: ", cfg.DATA.USE_HEATMAP)
    print("Using Text Aug: ", cfg.DATA.TEXT_AUG)
    if cfg.DATA.FRAMES_CONCAT or cfg.DATA.MULTI_FRAMES:
        print("Num frames: ", cfg.DATA.NUM_FRAMES)


    args.use_cuda = True
    train_data=CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH,
                                 transform=transform_train, motion_transform=motion_transform,
                                 frames_concat=cfg.DATA.FRAMES_CONCAT, use_multi_frames=cfg.DATA.MULTI_FRAMES)
    trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                             num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)
    # val_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH, transform=transform_test, Random=False)
    # valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE * 10, shuffle=False,
    #                        num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, )
    val_data_test = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test, val=True,
                                               frames_concat=cfg.DATA.FRAMES_CONCAT, use_multi_frames=cfg.DATA.MULTI_FRAMES)
    valloader = DataLoader(dataset=val_data_test, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)

    if cfg.MODEL.NUM_CLASS == 0:
        cfg.MODEL.NUM_CLASS = len(train_data)

    if args.eval_only:
        args.resume = True
    model = build_model(cfg, args)
    # print(model)

    model_freeze = FreezeBackbone(model, freeze_epoch=cfg.TRAIN.FREEZE_EPOCH)
    optimizer, scheduler = build_vanilla_optimizer(cfg, model, trainloader)

    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    else:
        assert False

    args.feat_idx = cfg.MODEL.MAIN_FEAT_IDX
    if args.eval_only:
        evaluate_by_test_all(model, valloader, 0, cfg, args.feat_idx, args, tokenizer, optimizer, multi_frames=cfg.DATA.MULTI_FRAMES)

        # if cfg.EVAL.ON2021:
        #     val_data_2021 = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test, val=True)
        #     valloader_2021 = DataLoader(dataset=val_data_test, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        #                            num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, )
        #     evaluate_by_test_all(model, valloader_2021, 0, cfg, args.feat_idx, args, tokenizer, optimizer)
        return

    start_time = time.monotonic()
    model.train()
    global_step = 0
    best_mrr = 0.

    metric_learning = cfg.MODEL.METRIC.LOSS
    if metric_learning == 'CircleLoss':
        print(f"====> Using metric loss: {metric_learning}")
        metric_loss_f = CircleLoss(m=cfg.MODEL.METRIC.LOSS_MARGIN, gamma=cfg.MODEL.METRIC.LOSS_SCALE)
    elif metric_learning == 'TripletLoss':
        print(f"====> Using metric loss: {metric_learning}")
        metric_loss_f = TripletLoss(margin=cfg.MODEL.METRIC.LOSS_MARGIN)
    elif metric_learning == 'PairCircleLoss':
        metric_loss_f = CirclePairLoss(s=cfg.MODEL.METRIC.LOSS_SCALE, m=cfg.MODEL.METRIC.LOSS_MARGIN)
    elif metric_learning == 'PairCosFace':
        metric_loss_f = CosFacePairLoss(s=cfg.MODEL.METRIC.LOSS_SCALE, m=cfg.MODEL.METRIC.LOSS_MARGIN)
    else:
        metric_loss_f = None
        metric_learning = 'None'

    model_freeze.start_freeze_backbone()
    for epoch in range(cfg.TRAIN.EPOCH):
        if cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (cfg.EVAL.EPOCH * cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
            evaluate_by_test(model, valloader, epoch, cfg, args.feat_idx, args, tokenizer, optimizer, cfg.DATA.MULTI_FRAMES)

        model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        clip_losses = AverageMeter('Clip_Loss', ':.4e')
        cls_losses = AverageMeter('Cls_Loss', ':.4e')
        metric_losses = AverageMeter(metric_learning, ':.4e')
        nlp_view_losses = AverageMeter(f'Nlp_View_{cfg.MODEL.HEAD.NLP_VIEW_LOSS}_Loss', ':.4e')
        mrr = AverageMeter('MRR', ':6.4f')
        top1_acc = AverageMeter('Acc@1', ':6.4f')
        top5_acc = AverageMeter('Acc@5', ':6.4f')
        progress = ProgressMeter(
            len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT, 
            [batch_time, data_time, losses, clip_losses, cls_losses, metric_losses, nlp_view_losses, mrr, top1_acc, top5_acc],
            prefix="Epoch: [{}]".format(epoch))
        end = time.time()
        epo_start_time = time.monotonic()
        for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
            model_freeze.on_train_epoch_start(epoch=epoch * cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
            for batch_idx, batch in enumerate(trainloader):
 
                image, text, car_text, view_text = batch["crop_data"], batch["text"], batch["car_text"], batch["view_text"]

                if cfg.DATA.USE_CLIP_FEATS:
                    clip_feats_text, clip_feats_vis = batch['clip_feats_text'], batch['clip_feats_vis']

                id_car, cam_id = batch["tmp_index"], batch["camera_id"]

                # same dual Text
                if cfg.MODEL.SAME_TEXT:
                    car_text = text

                if cfg.DATA.USE_MULTI_QUERIES:
                    all_input_ids = []
                    all_attn_masks = []
                    
                    #print(len(text), "BEFORE")
                    text = np.array(text).transpose(1, 0)
                    #print(len(text), "AFTER")
                    for txt in text:
                        txt_tokens = tokenizer.batch_encode_plus(txt, padding='longest', return_tensors='pt')
                        all_input_ids.append(txt_tokens['input_ids'].cuda())
                        all_attn_masks.append(txt_tokens['attention_mask'].cuda())
                    #print(len(all_attn_masks), "ATTN MASK")
                    #print(len(all_input_ids), "INPUT_IDS")
                else:
                    tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
                
                data_time.update(time.time() - end)
                global_step += 1
                optimizer.zero_grad()

                if cfg.DATA.USE_MOTION:
                    bk = batch["bk_data"]
                    if 'dual-text' in cfg.MODEL.NAME:
                        car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                        if 'view' in cfg.MODEL.NAME:
                            view_tokens = tokenizer.batch_encode_plus(view_text, padding='longest', return_tensors='pt')
                            outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda(),
                                            view_tokens['input_ids'].cuda(), view_tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                        else:
                            outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                    else:
                        # without dual text
                        if 'view' in cfg.MODEL.NAME:
                            view_tokens = tokenizer.batch_encode_plus(view_text, padding='longest', return_tensors='pt')


                            outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                        view_tokens['input_ids'].cuda(), view_tokens['attention_mask'].cuda(),
                                        image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                        else:
                            if cfg.DATA.USE_CLIP_FEATS:
                                outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                image.cuda(), bk.cuda(), targets=id_car.long().cuda(),
                                                clip_feats_text=clip_feats_text, clip_feats_vis=clip_feats_vis)
                            elif cfg.DATA.USE_MULTI_QUERIES:
                                outputs = model(all_input_ids, all_attn_masks,
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                            else:
                                outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                else:
                    # without motion
                    outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                    image.cuda(), targets=id_car.long().cuda())

                pairs, logit_scale, cls_logits = outputs['pairs'], outputs['logit_scale'], outputs['cls_logits']

                logit_scale = logit_scale.mean().exp()

                clip_loss = torch.tensor(0., device='cuda')
                acc_sim = 0.
                if cfg.MODEL.NAME == 'dual-text-add' and not cfg.MODEL.HEAD.ADD_TRAIN or \
                        cfg.MODEL.NAME == 'dual-text-cat' and not cfg.MODEL.HEAD.CAT_TRAIN:
                    loss_pairs = pairs[:-1]  # not train, remove merge feat
                else:
                    loss_pairs = pairs

                if cfg.MODEL.HEAD.CLIP_LOSS:
                    for visual_embeds, lang_embeds in loss_pairs:
                        sim_i_2_t = torch.matmul(visual_embeds, torch.t(lang_embeds))
                        acc_sim = sim_i_2_t.clone().detach()
                        sim_i_2_t = sim_i_2_t - (torch.eye(image.size(0)).cuda() * cfg.MODEL.HEAD.CLIP_LOSS_MARGIN)
                        sim_i_2_t = torch.mul(logit_scale, sim_i_2_t)
                        sim_t_2_i = sim_i_2_t.t()
                        loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
                        loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
                        clip_loss += (loss_t_2_i+loss_i_2_t)/2
                else:
                    acc_sim = torch.matmul(pairs[-1][0], torch.t(pairs[-1][1])).detach()

                nlp_view_loss = torch.tensor(0., device='cuda')
                if 'view' in cfg.MODEL.NAME:
                    view_lang_embeds, logit_scale_nl = outputs['view_nl']
                    logit_scale_nl = logit_scale_nl.mean().exp()
                    motion_vis_embeds, motion_lang_embeds = pairs[1]  # motion pair
                    if cfg.MODEL.HEAD.NLP_VIEW_LOSS == 'Triplet':
                        pos_sim = torch.diag(torch.matmul(motion_vis_embeds, motion_lang_embeds.T)).unsqueeze_(1)
                        neg_sim = torch.diag(torch.matmul(motion_vis_embeds, view_lang_embeds.T)).unsqueeze_(1)
                        delta = neg_sim - pos_sim + cfg.MODEL.HEAD.NLP_VIEW_LOSS_MARGIN
                        if cfg.MODEL.HEAD.NLP_VIEW_SOFT:
                            nlp_view_loss = F.softplus(logit_scale_nl * delta).mean()
                        else:
                            nlp_view_loss = F.relu(delta).mean()
                    else:
                        # contrastive
                        neg_sim = torch.diag(torch.matmul(motion_lang_embeds, view_lang_embeds.T)).unsqueeze_(1)
                        if cfg.MODEL.HEAD.NLP_VIEW_SOFT:
                            nlp_view_loss = F.softplus(logit_scale_nl * neg_sim).mean()
                        else:
                            nlp_view_loss = F.relu(neg_sim).mean()

                cls_loss = torch.tensor(0., device='cuda')
                for cls_logit in cls_logits:
                    cls_loss += cfg.MODEL.HEAD.CLS_WEIGHT * \
                            cross_entropy(cls_logit, id_car.long().cuda(), epsilon=cfg.MODEL.HEAD.CE_EPSILON)

                # metric learning
                metric_loss = torch.tensor(0., device='cuda')
                if metric_learning != 'None':
                    for pair in loss_pairs:
                        metric_loss += metric_loss_f(torch.cat(pair), torch.cat([id_car, id_car]).long().cuda())

                metric_loss *= cfg.MODEL.METRIC.METRIC_WEIGHT

                loss = clip_loss + cls_loss + metric_loss + nlp_view_loss

                acc1, acc5 = accuracy(acc_sim, torch.arange(image.size(0)).cuda(), topk=(1, 5))
                mrr_ = get_mrr(acc_sim)

                losses.update(loss.item(), image.size(0))
                clip_losses.update(clip_loss.item(), image.size(0))
                cls_losses.update(cls_loss.item(), image.size(0))
                metric_losses.update(metric_loss.item(), image.size(0))
                nlp_view_losses.update(nlp_view_loss.item(), image.size(0))
                mrr.update(mrr_.item() * 100, image.size(0))
                top1_acc.update(acc1[0], image.size(0))
                top5_acc.update(acc5[0], image.size(0))

                loss.backward()
                optimizer.step()

                scheduler.step()
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                    progress.display(global_step % (len(trainloader) * 30))

        if epoch % save_interval == 1:
            checkpoint_file = args.logs_dir + "/checkpoint_%d.pth" % epoch
            args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)

        if mrr.avg > best_mrr:
            best_mrr = mrr.avg
            checkpoint_file = args.logs_dir + "/checkpoint_best.pth"
            args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)

        epo_end_time = time.monotonic()
        print(f'Epoch {epoch} running time: ', timedelta(seconds=epo_end_time - epo_start_time))
        print(f'Logs dir: {args.logs_dir} ')

    del train_data, trainloader

    evaluate_by_test_all(model, valloader, cfg.TRAIN.EPOCH, cfg, args.feat_idx, args, tokenizer, optimizer, multi_frames=cfg.DATA.MULTI_FRAMES)

    # if cfg.EVAL.ON2021:
    #     val_data_2021 = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH_2021,
    #                                       transform=transform_test, Random=False, years=2021)
    #     valloader_2021 = DataLoader(dataset=val_data_2021, batch_size=cfg.TRAIN.BATCH_SIZE * 10, shuffle=False,
    #                                 num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=False, )
    #     evaluate_by_test_all(model, valloader_2021, cfg.TRAIN.EPOCH, cfg, args.feat_idx, args, tokenizer, optimizer)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    print(f'Logs dir: {args.logs_dir} ')


if __name__ == '__main__':
    main()
