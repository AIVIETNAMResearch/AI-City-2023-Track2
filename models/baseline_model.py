import torch.nn as nn
from frozen.frozen_model import FrozenInTime
import timm
from aggregation_head import ContextualizedWeightedHead
import torch

class VideoTextFeatureExtractor(nn.Module):
    def __init__(self, base_setting, text_head_setting, vision_head_setting, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.base = FrozenInTime(**base_setting, device=device)
        self.motion_encoder = FrozenInTime(**base_setting, device=device)
        self.motion_line_encoder = timm.create_model(vision_head_setting['motion_line_encoder'], num_classes=256, pretrained=True)
        
        self.text_head = ContextualizedWeightedHead(**text_head_setting['args'], device=device)
        self.vision_head = ContextualizedWeightedHead(**vision_head_setting['args'], device=device)
        
    def forward(self, frames, captions, motion, motion_line):
        frame_features, text_features = self.base.forward(frames, captions)
        motion_features = self.motion_encoder.compute_video(motion.unsqueeze(1))
        motion_line_features = self.motion_line_encoder(motion_line)
        
        vision_features = torch.stack([frame_features, motion_features, motion_line_features], dim=1)
        
        text_features = self.text_head(text_features)
        vision_features = self.vision_head(vision_features)
        
        return vision_features, text_features