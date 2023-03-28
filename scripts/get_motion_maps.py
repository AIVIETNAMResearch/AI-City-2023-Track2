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

#from config import BASE_DIR
BASE_DIR = '/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Track2-Vehicle-Retrieval/Track2-Vehicle_Retrieval'

# dataset_path = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval'
# data_path = '/data/datasets/aicity2021/AIC21_Track5_NL_Retrieval/new_baseline/AIC21_Track5_NL_Retrieval'

dataset_path = BASE_DIR + '/data/AIC23_Track2_NL_Retrieval/data'
data_path = BASE_DIR + '/data/AIC23_Track2_NL_Retrieval/data'

imgpath = dataset_path
# with open("data2021/test-tracks.json") as f:
#     tracks_test = json.load(f)
# with open("data2021/train-tracks.json") as f:
#     tracks_train = json.load(f)
with open("data/AIC23_Track2_NL_Retrieval/data/test-tracks.json") as f:
    tracks_test = json.load(f)
with open("data/AIC23_Track2_NL_Retrieval/data/train-tracks.json") as f:
    tracks_train = json.load(f)
all_tracks = tracks_train
for track in tracks_test:
    all_tracks[track] = tracks_test[track]
n_worker = 2

parser = argparse.ArgumentParser(description='motion image generator')
parser.add_argument('--iou', default=0.05, type=float,
                    help='iou_threshold')
parser.add_argument('--wo_bk', action='store_true',
                    help='don\'t use background image')
parser.add_argument('--use-frame', action='store_true',
                    help='select frame by time frame')
args = parser.parse_args()

iou_threshold = args.iou
with_bk = not args.wo_bk

save_bk_dir = osp.join(data_path, "data/bk_map")
os.makedirs(save_bk_dir, exist_ok=True)
save_mo_dir = osp.join(data_path, "data/motion_map")
os.makedirs(save_mo_dir, exist_ok=True)
iou_path = f"data/motion_map_iou" if with_bk else f"data/motion_map_iou_nobk"
save_mo_iou_dir = osp.join(data_path, iou_path)
os.makedirs(save_mo_iou_dir, exist_ok=True)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def get_bk_map(info):
    path, save_name = info
    img = glob.glob(path + "/img1/*")
    img.sort()
    interval = min(5, max(1, int(len(img)/200)))
    img = img[::interval][:1000]
    imgs = []
    for name in img:
        # print(name)
        imgs.append(cv2.imread(name))
    avg_img = np.mean(np.stack(imgs), 0)
    avg_img = avg_img.astype(np.int)
    img_name = save_bk_dir + "/%s.jpg" % save_name
    cv2.imwrite(img_name, avg_img)
    print(f"background: {img_name} save done")
    return path, avg_img.shape, name


def get_motion_map(info):
    track, track_id = info
    for i in range(len(track["frames"])):
        frame_path = track["frames"][i]
        frame_path = os.path.join(imgpath, frame_path)
        frame = cv2.imread(frame_path)
        box = track["boxes"][i]
        if i == 0:
            example = np.zeros(frame.shape, np.int)
        if i % 7 == 1:
            example[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :] = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]

    words = track["frames"][0].split('/')
    bk_path = osp.join(data_path, "data/bk_map", words[-4] + '_' + words[-3] + '.jpg')
    #print(bk_path)
    avg_img = cv2.imread(bk_path).astype(np.int)
    postions = (example[:, :, 0] == 0) & (example[:, :, 1] == 0) & (example[:, :, 2] == 0)
    example[postions] = avg_img[postions]
    img_name = save_mo_dir + "/%s.jpg" % track_id
    cv2.imwrite(img_name, example)
    print(f"motion: {img_name} save done")


def get_motion_map_iou(info):
    track, track_id = info


    first_box = track["boxes"][0]
    first_area = first_box[2] * first_box[3]
    last_box = track["boxes"][-1]
    last_area = last_box[2] * last_box[3]

    use_bk = True
    area_first = False

    cnt_list = range(len(track["frames"]))
    if area_first:
        range_list = cnt_list if first_area > last_area else cnt_list[::-1]
    else:
        range_list = cnt_list

    prev_rect = None
    for cnt, i in zip(cnt_list, range_list):
        frame_path = track["frames"][i]
        frame_path = os.path.join(imgpath, frame_path[2:])
        #print(frame_path)
        frame = cv2.imread(frame_path)
        # box = np.array(box)
        box = np.array(track["boxes"][i])
        def f(rec):
            # convert to (top, left, bottom, right)
            rec[2] = rec[2] + rec[0]
            rec[3] = rec[3] + rec[1]
            return rec
        rect = f(box.copy())
        if cnt == 0:
            example = np.zeros(frame.shape, np.int)
            prev_rect = rect
            if with_bk:
                example[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :] = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
            else:
                example = frame
        else:
            iou = compute_iou(prev_rect, rect)
            if iou <= iou_threshold:
                prev_rect = rect
                example[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :] = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
    if with_bk:
        words = track["frames"][0].split('/')
        bk_path = osp.join(data_path, "data/bk_map", words[-4] + '_' + words[-3] + '.jpg')
        avg_img = cv2.imread(bk_path).astype(np.int)
        postions = (example[:, :, 0] == 0) & (example[:, :, 1] == 0) & (example[:, :, 2] == 0)
        example[postions] = avg_img[postions]
    img_name = save_mo_iou_dir + "/%s.jpg" % track_id
    #print(img_name)
    cv2.imwrite(img_name, example)
    print(f"motion: {img_name} save done")

root = dataset_path
# paths = ["train/S01", "train/S03", "train/S04", "validation/S02", "validation/S05"]
paths = ['train/S01', 'validation/S02', 'train/S03', 'train/S04', 'validation/S05', ]
files = []
for path in paths:
    seq_list = os.listdir(osp.join(root, path))
    for seq in seq_list:
        files.append((os.path.join(root, path, seq), path[-3:] + '_' + seq))
#print(files)
#produce background image
# with multiprocessing.Pool(n_worker) as pool:
#     # get_bk_map(files)
#     for imgs in tqdm(pool.imap_unordered(get_bk_map, files)):
#         pass

all_tracks_ids = list(all_tracks.keys())
files = []
for track_id in all_tracks:
    files.append((all_tracks[track_id], track_id))

if args.use_frame:
    # get_motion_map(files)
    with multiprocessing.Pool(n_worker) as pool:
        for imgs in tqdm(pool.imap_unordered(get_motion_map, files)):
            pass
else:
    # iou
    # get_motion_map_iou(files)
    #print(files)

    with multiprocessing.Pool(n_worker) as pool:
        for imgs in tqdm(pool.imap_unordered(get_motion_map_iou, files)):
            pass


