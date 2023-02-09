import json
import numpy as np
from tqdm import tqdm

from preprocessing.image_encoder import ImageEncoder
from preprocessing.process import Track2CustomDataset
from preprocessing.process import get_image_context_bbox, get_motion_img, get_transforms, bb_intersection_over_union
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    with open("train_tracks.json", "r") as f:
        data = json.load(f)
    image_encoder = ImageEncoder()
    
    dataset = Track2CustomDataset(data, transforms= get_transforms(224, train=True))
    sample = dataset[0]
    print(sample['image'].shape, sample['context'].shape, sample['context'].shape, sample['motion_line'].shape)

    dataloader = DataLoader(dataset, batch_size =1, shuffle=False, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        with torch.no_grad():
            image_embeddings, context_embeddings, motion_embeddings, motion_line_embeddings = image_encoder(sample_batched)
        break
    
    print(image_embeddings.shape, context_embeddings.shape, motion_embeddings.shape, motion_line_embeddings.shape)
    
    
