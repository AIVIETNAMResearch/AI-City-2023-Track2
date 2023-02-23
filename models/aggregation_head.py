import torch.nn.functional as F
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, dim_list, last_nonlinear=False, layer_norm=False):
        super().__init__()

        if len(dim_list) >= 2:
            layers = []
            for i in range(len(dim_list) - 1):
                layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim_list[i+1]))
                layers.append(nn.ReLU())
            if not last_nonlinear:
                layers = layers[:-1]
                if layer_norm:
                    layers = layers[:-1]
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()
    def forward(self, x):
        return self.model(x)

class ContextualizedWeightedHead(nn.Module):
    def __init__(self, d_model, nhead, num_layers, fc_dim_list, temperature=1.,device=torch.device('cpu'), layer_norm=False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = FCNet(fc_dim_list, last_nonlinear=False, layer_norm=layer_norm)
        self.temperature = temperature

    def forward(self, features):
        attention_out = self.attention(features)
        weight = self.head(attention_out)
        weight = F.softmax(weight / self.temperature, dim=1)
        weighted_features = features * weight
        return weighted_features.mean(dim=1)

    def gen_weights(self, features):
        attention_out = self.attention(features)
        weight = self.head(attention_out)
        weight = F.softmax(weight / self.temperature, dim=1)
        return weight