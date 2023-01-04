import os
import os.path as osp
import json
import random
random.seed(888)

# track = 'data2021'
track = 'data2022'

with open(osp.join(track, "train-tracks.json")) as f:
    tracks_train = json.load(f)

keys = list(tracks_train.keys())
random.shuffle(keys)
train_data = dict()
val_data = dict()
for key in keys[:100]:
    val_data[key] = tracks_train[key]
for key in keys[100:]:
    train_data[key] = tracks_train[key]
with open(osp.join(track, "train.json"), "w") as f:
    json.dump(train_data, f, indent=4)
with open(osp.join(track, "val.json"), "w") as f:
    json.dump(val_data, f, indent=4)
