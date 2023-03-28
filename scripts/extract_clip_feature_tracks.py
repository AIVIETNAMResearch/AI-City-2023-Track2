import os, shutil
import torch
import glob
import json
import clip
from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel, CLIPTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, preprocess = clip.load("ViT-B/16", device=device)

dst_path = "../data/AIC23_Track2_NL_Retrieval/data/tracks"
shutil.rmtree(dst_path, ignore_errors=True)
os.makedirs(dst_path, exist_ok = True)

def extract_clip_feature(tracks_path):
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
        # print(list(tracks.keys()))
        print(len(tracks))
        frame_paths = []
        choose_boxes = []
        features = []
        
        for track in tracks:
            dst = f'{dst_path}/{track}'
            # os.makedirs(dst, exist_ok = True)
            frames = tracks[track]["frames"]
            boxes  = tracks[track]["boxes"]

            max_area = min_area = 0
            max_area_box = []
            for i, box in enumerate(boxes):
                area = box[2]*box[3]

                # if i == 0:
                #     min_area = area
                
                # if area < min_area:
                #     min_area = area
                #     min_area_idx = i

                if area > max_area:
                    max_area = area
                    max_area_idx = i
                    max_area_box = box

            frame_path = frames[max_area_idx]
            frame_path = "../data/AIC23_Track2_NL_Retrieval/data" + frame_path[1:]
            frame = Image.open(frame_path)
            box = boxes[max_area_idx]
            fr = frame.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
            # fr.save(f'{dst}.jpg')

            # Extract features
            image = preprocess(fr).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features.detach().cpu()
            # print(image_features.shape)

            features.append(image_features)
            frame_paths.append(frame_path)
            choose_boxes.append(max_area_box)

            data = {
            "tracks": list(tracks.keys()),
            "frames": frames,
            "boxes": choose_boxes,
            "features": features
        }

    name_tracks = tracks_path.split("/")[-1].split(".")[0]
    torch.save(data, f"../data/AIC23_Track2_NL_Retrieval/data/{name_tracks}.pt")

if __name__ == "__main__":
    tracks = ["../data/AIC23_Track2_NL_Retrieval/data/train-tracks.json",
              "../data/AIC23_Track2_NL_Retrieval/data/test-tracks.json"]
    
    for track_path in tracks:
        extract_clip_feature(track_path)