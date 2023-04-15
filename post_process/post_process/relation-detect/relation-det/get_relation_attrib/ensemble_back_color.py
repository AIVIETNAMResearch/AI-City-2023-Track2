import numpy as np
import json

fea_dic = np.load("./back_color.npy",allow_pickle=True).item()

color_dic = {}
for i in fea_dic:
    logits = np.zeros(8)
    for prob in fea_dic[i]:
        logits += prob[0][0]
    logits = logits / max(len(fea_dic[i]),1)
    
    if len(fea_dic[i]) > 5:
        pred = np.argmax(logits)
        score = logits[pred]
    else:
        pred = -1
        score = -1 
        logits = np.zeros(8)
    color_dic[i] = logits
    
    
with open("./back_test_color_predict_no_avg_0.5.json") as f:
# with open("./back_test_color_predict_one_hot_0.5.json") as f:
    data_color = json.load(f)
    
track_color = {}

for key in data_color:
    dic_color = {}
    color_list = data_color[key]
    count_tot = 0
    valid = 0
    color = np.array((-1))
    conf_mat = np.zeros((8))
    for i in range(len(color_list)):
        if color_list[i] != -1:
            pred_mat = np.array(color_list[i])
            conf_mat = conf_mat + pred_mat
            valid += 1
            
    conf_mat = conf_mat / max(1,valid)
    if valid > 5:
        pred = np.argmax(conf_mat)
        if conf_mat[pred] > 0:
            color = pred
    else:
        conf_mat = np.zeros((8))
    track_color[key] = conf_mat

    
color_new = {}
for i in color_dic:
    if color_dic[i].sum() > 0 and track_color[i].sum() > 0:
        logits = (color_dic[i] + track_color[i]) /2
        pred = np.argmax(logits)
        score = logits[pred]
        color_new[i] = [pred,score]
    else:
        color_new[i] = [-1,-1]
        
        
np.save("./back_color_ensemble.npy",color_new)