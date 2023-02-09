import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    efficientnet = timm.create_model('efficientnet_b4', pretrained=True).to(device)
    def __init__(self, image_encoder=efficientnet, context_encoder=efficientnet, 
                 motion_encoder=efficientnet, motion_line_encoder=efficientnet):
        super(ImageEncoder, self).__init__()

        self.image_encoder = image_encoder
        self.context_encoder = context_encoder
        self.motion_encoder = motion_encoder
        self.motion_line_encoder = motion_line_encoder

    def forward(self, input):
        # print(input)
        image_features = self.image_encoder(input['image'])
        context_features = self.context_encoder(input['context'])
        motion_features = self.motion_encoder(input['motion'])
        motion_line_features = self.motion_line_encoder(input['motion_line'])

        return image_features, context_features, motion_features, motion_line_features