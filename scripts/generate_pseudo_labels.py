import json
import os 
import sys
import uuid
import cv2

import sys
sys.path.append('/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Track2-Vehicle-Retrieval/Track2-Vehicle_Retrieval')

from preprocessing.transforms import build_transforms, build_vanilla_transforms, build_motion_transform, BackTranslateAug
trans_aug = BackTranslateAug()

with open("data/AIC23_Track2_NL_Retrieval/data/test-tracks.json") as f:
    test_tracks = json.load(f)

with open("data/AIC23_Track2_NL_Retrieval/data/test-queries.json") as f:
    test_queries = json.load(f)

with open('submissions/submit_10_45ath.json') as f:
    test_results = json.load(f)


pseudo_labels = {}
for i, (key, val) in enumerate(test_results.items()):
    print(40*"-")
    query = test_queries[key]
    nls = query['nl']
    print(nls)
    texts_aug = [trans_aug(nl) for nl in nls]
    print(texts_aug)
    if i > 5:
        break