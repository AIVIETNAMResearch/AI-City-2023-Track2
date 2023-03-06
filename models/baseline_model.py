import torch.nn as nn
from .frozen.frozen_model import FrozenInTime
import timm
from .aggregation_head import ContextualizedWeightedHead, FCNet
import torch
import torch.nn.functional as F

class VideoTextFeatureExtractor(nn.Module):
    def __init__(self, base_setting, text_head_setting, vision_head_setting, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.base = FrozenInTime(**base_setting, device=device)
        self.motion_encoder = FrozenInTime(**base_setting, device=device)
        self.motion_line_encoder = timm.create_model(vision_head_setting['motion_line_encoder'], num_classes=500, pretrained=False)
        
        self.color_fc = FCNet([256, 512,256], layer_norm=True)
        self.type_fc = FCNet([256, 512, 256], layer_norm=True)
        self.motion_fc = FCNet([256, 512, 256], layer_norm=True)
        self.motion_line_fc = FCNet([500, 512, 256], layer_norm=True)

        self.color_fc2 = FCNet([256, base_setting['video_params']['color_classes']])
        self.type_fc2 = FCNet([256, base_setting['video_params']['type_classes']])
        self.motion_fc2 = FCNet([256, base_setting['video_params']['motion_classes']])
        #self.motion_line_fc2 = FCNet([256, base_setting['video_params']['motion_classes']])

        self.text_head = ContextualizedWeightedHead(**text_head_setting['args'], device=device)
        self.vision_head = ContextualizedWeightedHead(**vision_head_setting['args'], device=device)
        
    def forward(self, frames, captions, motion, motion_line):
        bz = frames.shape[0]

        text_features = self.compute_text(captions, bz)
        vision_features, color_features, type_features, motion_features = self.compute_video(frames, motion, motion_line)
        
        color_logits = F.softmax(self.color_fc2(color_features))
        type_logits = F.softmax(self.type_fc2(type_features))
        motion_logits = F.softmax(self.motion_fc2(motion_features))

        return vision_features, text_features, color_logits, type_logits, motion_logits

    def compute_video(self, frames, motion, motion_line):
        frame_features = self.base.compute_video(frames)
        color_features = self.color_fc(frame_features)
        type_features = self.type_fc(frame_features)

        motion_features = self.motion_encoder.compute_video(motion.unsqueeze(1))
        motion_features = self.motion_fc(motion_features)

        motion_line_features = self.motion_line_encoder(motion_line)
        motion_line_features = self.motion_line_fc(motion_line_features)

        vision_features = torch.stack([frame_features, color_features, type_features, motion_features, motion_line_features], dim=1)
        vision_features = self.vision_head(vision_features)
        return vision_features, color_features, type_features, motion_line_features
    
    def compute_text(self, captions):
        text_features = self.base.compute_text(captions)
        text_features = self.text_head(text_features)
        return text_features
    
    def compute_text(self, captions, batch_size):
        self.base.bz = batch_size
        text_features = self.base.compute_text(captions)
        text_features = self.text_head(text_features)
        return text_features