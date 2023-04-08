from yacs.config import CfgNode as CN
import os
import os.path as osp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_C = CN()

dataset_path_2021 = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval'
data_path_2021 = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval/new_baseline/AIC21_Track5_NL_Retrieval'
save_mo_dir_2021 = osp.join(data_path_2021, "data/motion_map_iou")

dataset_path = './data/AIC23_Track2_NL_Retrieval/data'
data_path = './data/AIC23_Track2_NL_Retrieval/data'
save_mo_dir = osp.join(data_path, "data/motion_map_iou")
save_heatmap_dir = osp.join(data_path, "data/motion_heatmap")
save_clip_feats = osp.join(data_path, "clip_feats")


# DATA process related configurations.
_C.DATA = CN()
_C.DATA.CITYFLOW_PATH = dataset_path
_C.DATA.NUM_FRAMES = 2
# _C.DATA.TRAIN_JSON_PATH = "data/train.json"
# _C.DATA.EVAL_JSON_PATH = "data/val.json"
_C.DATA.TRAIN_JSON_PATH = "data/train.json"
_C.DATA.EVAL_JSON_PATH = "data/val.json"
_C.DATA.EVAL_JSON_PATH_2021 = "data2021/train-tracks.json"
_C.DATA.SIZE = 288
_C.DATA.FRAMES_CONCAT = False
_C.DATA.CROP_AREA = 1. ## new_w = CROP_AREA * old_w
# _C.DATA.TEST_TRACKS_JSON_PATH = "data/test-tracks.json"
_C.DATA.TEST_TRACKS_JSON_PATH = "data/AIC23_Track2_NL_Retrieval/data/test-tracks.json"
_C.DATA.TEST_OUTPUT = "logs/Test_Output"
_C.DATA.USE_MOTION = True
_C.DATA.MOTION_PATH = save_mo_dir
_C.DATA.MOTION_PATH_2021 = save_mo_dir_2021
_C.DATA.CLIP_PATH = save_clip_feats
_C.DATA.USE_HEATMAP = False
_C.DATA.HEATMAP_PATH = save_heatmap_dir
_C.DATA.MULTI_FRAMES = False
_C.DATA.TEXT_AUG = False
_C.DATA.USE_CLIP_FEATS = False
_C.DATA.USE_MULTI_QUERIES = False
# _C.DATA.MOTION_PATH = "data/motion_map"
_C.DATA.USE_OSS = False  # set True if using OSS
_C.DATA.OSS_PATH = 's3://chenhaobo-shared'
_C.DATA.CROP_AUG = False
_C.DATA.TEXT_AUG_SPLIT = ''


# Model specific configurations.
_C.MODEL = CN()

_C.MODEL.NAME = "base" #base or dual-stream
_C.MODEL.BERT_TYPE = "BERT"
_C.MODEL.BERT_NAME = "roberta-base"
_C.MODEL.IMG_ENCODER = "efficientnet-b2" # "se_resnext50_32x4d" # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3
# _C.MODEL.NUM_CLASS = 2498
_C.MODEL.NUM_CLASS = 2155  # 2155
_C.MODEL.OUTPUT_SIZE = 1024
_C.MODEL.EMBED_DIM = 1024
_C.MODEL.car_idloss = True
_C.MODEL.mo_idloss = True
_C.MODEL.share_idloss = True
_C.MODEL.con_idloss = True
_C.MODEL.RESNET_CHECKPOINT = osp.join(os.environ['HOME'], '.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')

# dual Text
_C.MODEL.SAME_TEXT = False
_C.MODEL.MERGE_DIM = 2048

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.ADD_TRAIN = True
_C.MODEL.HEAD.CAT_TRAIN = True
_C.MODEL.HEAD.CLS_NONLINEAR = True
_C.MODEL.HEAD.CLS_WEIGHT = 0.5
_C.MODEL.HEAD.CE_EPSILON = 0.
_C.MODEL.HEAD.CLS_LOSS_SCALE = 32
_C.MODEL.HEAD.CLS_LOSS_MARGIN = 0.35
_C.MODEL.HEAD.SHARED_CLS = 'Linear'
_C.MODEL.HEAD.CAR_CLS = 'Linear'
_C.MODEL.HEAD.MO_CLS = 'Linear'

_C.MODEL.HEAD.CLIP_LOSS = True
_C.MODEL.HEAD.CLIP_LOSS_MARGIN = 0.

_C.MODEL.HEAD.NLP_VIEW_LOSS = 'Triplet'
_C.MODEL.HEAD.NLP_VIEW_LOSS_MARGIN = 0.
_C.MODEL.HEAD.NLP_VIEW_SOFT = True

_C.MODEL.METRIC = CN()
_C.MODEL.METRIC.METRIC_WEIGHT = 1.
_C.MODEL.METRIC.LOSS = ''
_C.MODEL.METRIC.LOSS_SCALE = 80
_C.MODEL.METRIC.LOSS_MARGIN = 0.4

_C.MODEL.MAIN_FEAT_IDX = -1

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.ONE_EPOCH_REPEAT = 1
_C.TRAIN.EPOCH = 400
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.LR = CN()
_C.TRAIN.LR.BASE_LR = 0.01
_C.TRAIN.LR.WARMUP_EPOCH = 40
_C.TRAIN.LR.DELAY = 120
_C.TRAIN.BENCHMARK = False
# _C.TRAIN.BENCHMARK = True  # set False for final training version
_C.TRAIN.FREEZE_EPOCH = 0
_C.TRAIN.SEED = 42
_C.TRAIN.DETERMINISTIC = False

# Test configurations
_C.TEST = CN()
_C.TEST.RESTORE_FROM = None
# _C.TEST.QUERY_JSON_PATH = "data/test-queries.json"
_C.TEST.QUERY_JSON_PATH = "data/test-queries.json"
_C.TEST.BATCH_SIZE = 128
_C.TEST.NUM_WORKERS = 6
_C.TEST.CONTINUE = ""
_C.TEST.RERANK = False


_C.EVAL = CN()
_C.EVAL.EPOCH = 20
_C.EVAL.EVAL_BY_TEST = True
_C.EVAL.EVAL_BY_TEST_NUM = 1  # eval_by_test epo = num * EVAL.EPOCH
_C.EVAL.ON2021 = False
# _C.EVAL.ON2021 = True

# -----------------------------------------------------------------------------
# INPUT from AICITY2021 DUN
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [288, 288]
# Size of the image during test
_C.INPUT.SIZE_TEST = [288, 288]
_C.INPUT.COLORJIT_PROB = 1.0
_C.INPUT.AUGMIX_PROB = 0.0
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
_C.INPUT.RE_SH = 0.4
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# color space
_C.INPUT.COLOR_SPACE = 'rgb'
# random patch
_C.INPUT.RANDOM_PATCH_PROB = 0.0
# random affine
_C.INPUT.RANDOM_AFFINE_PROB = 0.0
_C.INPUT.VERTICAL_FLIP_PROB = 0.0
# random blur
_C.INPUT.RANDOM_BLUR_PROB = 0.0

# cut-off long-tailed data
_C.INPUT.CUTOFF_LONGTAILED = False
_C.INPUT.LONGTAILED_THR = 2


def get_default_config():
    return _C.clone()