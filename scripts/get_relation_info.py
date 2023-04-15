import os
import cv2
import collections
import json
import matplotlib.pyplot as plt
import spacy
import pickle
import numpy as np
import random
nlp = spacy.load("en_core_web_sm")

# frame_id: S01_c001_000001
root = './data/AIC23_Track2_NL_Retrieval/data/'
det_dict = collections.defaultdict(list)
for sub_dir in ['train/S01', 'validation/S02', 'train/S03', 'train/S04', 'validation/S05']:
    sid = sub_dir.split('/')[-1]
    for cid in os.listdir(os.path.join(root, sub_dir)):
        train_path = os.path.join(os.path.join(root, sub_dir), cid)
        # det_path = os.path.join(train_path, 'det/det_mask_rcnn.txt')
        det_path = os.path.join(train_path, 'det/det_yolo3.txt')
        # det_path = os.path.join(train_path, 'det/det_ssd512.txt')

        det_results = []

        with open(det_path, 'r') as fb:

            for line in fb.readlines():
                frame, ID, left, top, width, height, _, _, _, _ = line.split(',')
                frame_id = '%s_%s_%s' % (sid, cid, int(frame))

                ID = int(ID)
                left, top, width, height = map(lambda x: int(float(x)), (left, top, width, height))

                det_results.append([frame, ID, left, top, width, height])
                det_dict[frame_id].append([left, top, width, height])

with open("data/AIC23_Track2_NL_Retrieval/data/test_nlpaug.json", 'r') as fb:
    test_tracks = json.load(fb)

with open("data/AIC23_Track2_NL_Retrieval/data/test_nlpaug.json", 'r') as fb:
    test_queries = json.load(fb)

print('length of test_tracks: %d' % len(test_tracks))
print('length of test_queries: %d' % len(test_queries))


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


'''
test_track_bboxes:
{
    "track_id": {
        "frame_1": [gt_box, ...],
        "frame_2": [gt_box, ...],
        ...
    }
}
First box of each track_id is the gt_box of this track
'''
test_track_bboxes = dict()
num_samples = 3
for track_id, track in test_tracks.items():
    test_track_bboxes[track_id] = dict()
    ids = random.choices(track['frames'], k=num_samples)
    #     print(ids)

    for idx in ids:
        test_track_bboxes[track_id][idx] = []
        frame = idx
        # left, top, width, height
        for box, frame_name in zip(track['boxes'], track['frames']):
            if frame_name == frame:
                gt_box = box
                break
        #         gt_box = track['boxes'][idx]
        cid = frame.split('/')[-3]
        sid = frame.split('/')[-4]
        frame_num = int(frame.split('/')[-1].split('.')[-2])
        frame_id = '%s_%s_%s' % (sid, cid, frame_num)
        all_boxes = det_dict[frame_id]

        # left, top, width, height
        #         image = cv2.imread(os.path.join('/data/datasets/aicity2022/track2', frame))
        #         cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
        max_iou = 0
        max_iou_id = 0
        for iou_id, box in enumerate(all_boxes):
            box1 = (gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3])
            box2 = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            #             cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
            iou = compute_iou(box1, box2)
            if iou > max_iou:
                max_iou = iou
                max_iou_id = iou_id

        for iou_id, box in enumerate(all_boxes):
            if box[2] * box[3] >= (gt_box[2] * gt_box[3] * 2 / 3) and iou_id != max_iou_id:
                test_track_bboxes[track_id][idx].append(box)

        test_track_bboxes[track_id][idx].insert(0, gt_box)


with open('test_track_bboxes.json', 'w') as fb:
    json.dump(test_track_bboxes, fb)


all_cars = set()
for _, query in test_queries.items():
    texts = query["nl"]
    for text in texts:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            all_cars.add(chunk.root.text.lower())

query_car_dict = collections.defaultdict(list)

for query_id, query in test_queries.items():
    texts = query["nl"]
    cars = [[] for _ in range(len(texts))]
    for idx, text in enumerate(texts):
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            for word in str(chunk).split(' '):
                if word.lower() in all_cars:
                    cars[idx].append(str(chunk).lower())
                    break
    cars = sorted(cars, key=lambda x: -len(x))
    query_car_dict[query_id] = cars[0]

with open('test_query_cars.json', 'w') as fb:
    json.dump(query_car_dict, fb)

# with open('/data/AIC21-R1/data/query_lang_embeds.pkl', 'rb') as fb:
#     query_lang_embeds = pickle.load(fb)

# with open('/data/AIC21-R1/data/track_car_embeds.pkl', 'rb') as fb:
#     track_car_embeds = pickle.load(fb)