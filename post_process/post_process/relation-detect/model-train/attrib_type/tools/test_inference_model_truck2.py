"""
test truck and cal acc at different thres
"""
import paddle
import paddle.fluid as fluid

import os

import numpy as np
import pdb
import argparse
import time
import pickle
import cv2

paddle.enable_static()


img_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
img_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)


#weight_dir = '/home/vis/jiangminyue/jiangminyue-carrec2/infer_weights/Res2Net50_model_iter959999'
#weight_dir = '/home/vis/jiangminyue/jiangminyue-carrec2/infer_weights/Res2Net50_model_iter1109999'
weight_dir = '/home/vis/jiangminyue/carrec-dynamic/inference'
img_dir = '/home/vis/jiangminyue/jiangminyue-carrec2/data/truck' 

gts = {}
gt_list = '/home/vis/jiangminyue/jiangminyue-carrec2/data/truck/truck_second_hand_test_img_list.txt'
all_imgs = []
with open(gt_list, 'r') as fid:
    for line in fid.readlines():
        name, label = line.strip().split(' ')
        gts[name]=label
        all_imgs.append(name)


chinese_tag = {}
tag_list = "/home/vis/jiangminyue/dataset/truck/truck_tag.list"
with open(tag_list, 'r') as fid:
    for line in fid.readlines():
        tag_index, name = line.strip().split(' ')
        chinese_tag[int(tag_index)-1]=name


place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

inference_program, feed_target_names, fetch_targets = \
        fluid.io.load_inference_model(weight_dir, exe, model_filename='model', params_filename='params')

# decode img
# resize
# normalize
# permute

time_count =0

thres = 0.7
count = 0
thres_count = 0
start_time = time.time()
eval_list = []
for each in all_imgs:
    #if '/133/' not in each:
    #    continue
    time_count += 1
    if time_count == 100:
        print(time.time()-start_time)
    img_path = os.path.join(img_dir, each)
    #print(img_path)
    #print(time_count, img_path)
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224))
    im = im.astype(np.float32, copy=False)
    im = im / 255.0
    #pdb.set_trace()
    im = im - np.array([0.485, 0.456, 0.406], dtype='float32')
    im = im / np.array([0.229, 0.224, 0.225], dtype='float32')
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    fea = exe.run(inference_program, fetch_list= fetch_targets, \
            feed = {feed_target_names[0]:im[np.newaxis, :]}, return_numpy=True)
    pred = fea[0].argmax()
    pred_score = fea[0].max()
    cur_label = int(gts[each])
    if pred == cur_label:
        count += 1
    else:
        print('img:{}, pred:{}, gt:{}'.format(each, chinese_tag[pred], chinese_tag[cur_label]))

    eval_list.append([pred, cur_label, pred_score])
    #pdb.set_trace()


    #if fea[0].max() > thres:
    #    thres_count += 1
    #
    #    cur_label = int(gts[each])
    #    #pdb.set_trace()
    #    if pred == cur_label:
    #        count +=1
    #    else:
    #        print('img:{}, pred:{}, gt:{}'.format(each, chinese_tag[pred], chinese_tag[cur_label]))
    #    #pdb.set_trace()
#print(count)
#print(thres_count)
#print(float(count) / thres_count)
test_thres = [i for i in range(50, 100, 5)]
test_thres = [0.0] + test_thres
for i in test_thres:
    thres = i / 100.0
    count = 0
    thres_count = 0
    for each in eval_list:
        cur_pred, cur_label, cur_score = each
        if cur_score > thres:
            thres_count += 1
            if cur_pred == cur_label:
                count += 1
    print("thres: {}, correct/pred: {}/{}, precision:{}".format(thres, count, thres_count, float(count)/thres_count))
