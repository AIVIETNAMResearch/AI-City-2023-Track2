import json
import os
import sys

import cv2
import numpy as np

sys.path.append('./EfficientNet-PyTorch')

import PIL
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import datasets, models, transforms

IMAGE_SIZE = (224,224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class VehicleClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=True)
        backbone._fc =  nn.Identity()
      
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )
        
    def extract_feature(self, input):
        x = self.feature_extractor(input)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
        logits = torch.softmax(logits, dim=-1)
        return logits

class ColorClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=True)
        backbone._fc =  nn.Identity()
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )
        
    def extract_feature(self, input):
        x = self.feature_extractor(input)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
        return logits

def get_state_dict(weight_path):
    state_dict = torch.load(weight_path)
    return state_dict



class DirectionClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = EfficientNet.from_pretrained(cfg['MODEL'], num_classes=cfg['NUM_CLASSES'], include_top=True)
        backbone._fc =  nn.Identity()
        out_channel = backbone._conv_head.out_channels

        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_channel, cfg['NUM_CLASSES'])
        )
        
    def extract_feature(self, input):
        x = self.feature_extractor(input)
        return x
        
    def forward(self, input):
        x = self.extract_feature(input)
        logits = self.classifier(x)
       
        return logits

def get_state_dict(weight_path):
    state_dict = torch.load(weight_path)
    return state_dict

def init_model(cfg_veh, cfg_col, cfg_dir,load_ckpt=True):
    veh_model = VehicleClassifier(cfg_veh)
    col_model = ColorClassifier(cfg_col)
    dir_model = DirectionClassifier(cfg_dir)
    veh_model.eval()
    col_model.eval()

    if load_ckpt:
        veh_weight = cfg_veh['WEIGHT']
        col_weight = cfg_col['WEIGHT']
        dir_weight = cfg_dir['WEIGHT']

        veh_model.load_state_dict(get_state_dict(veh_weight),strict=False) 
        col_model.load_state_dict(get_state_dict(col_weight),strict=False) 
        dir_model.load_state_dict(get_state_dict(dir_weight),strict=False)    


    return veh_model, col_model, dir_model
    
def preprocess_input(img):
    img = img.convert('RGB')
    img = val_transform(img)
    return img
