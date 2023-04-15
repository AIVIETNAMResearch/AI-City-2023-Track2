import paddle
import paddle.fluid as fluid
import os
from tqdm import tqdm
import numpy as np
import pickle
import random
from glob import glob
import paddle.nn.functional as F
import cv2
import json

paddle.enable_static()

weight_dir = './inference/'
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(weight_dir, exe, 
                                                model_filename='inference.pdmodel', params_filename='inference.pdiparams')


# all_img = []
# with open("/ssd2/yuyue/AICITY2022/data/label/test_color_aic22.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         path,label = line.strip("\n").split("\t")
#         all_img.append([path,label])
        
all_img = []
with open("/ssd2/yuyue/AICITY2022/data/aic22/test_tracks.json") as f:
    data = json.load(f)
for key in data.keys():
    paths = data[key]['frames']
    for path in paths:
        img_path = os.path.join("/ssd2/yuyue/AICITY2022/data/aic22_crop/",path.replace("./",""))
        img_path = os.path.dirname(img_path) + "/"+ key + "/" + img_path.split("/")[-1]
        assert os.path.exists(img_path)
        all_img.append(img_path)
        
fea_dic = {}

correct = 0
for i in tqdm(all_img):
    path = i
    name = path.split("/")[-2] + "_" + path.split("/")[-1]
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224))
    im = im.astype(np.float32, copy=False)
    im = im / 255.0
    im = im - np.array([0.485, 0.456, 0.406], dtype='float32')
    im = im / np.array([0.229, 0.224, 0.225], dtype='float32')
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    
    output,fea = exe.run(inference_program, fetch_list= fetch_targets, 
                                 feed = {feed_target_names[0]:im[np.newaxis,:]},
                                 return_numpy=True)
    
    fea_dic[name] = fea

with open("color_fea.pkl","wb") as f:
    pickle.dump(fea_dic,f)

correct