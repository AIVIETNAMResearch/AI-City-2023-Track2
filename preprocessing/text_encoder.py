from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class TextFeatureExtractor(nn.Module):
    def __init__(self, backbone, out_dim):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.mean_pooling = MeanPooling()
        self.fc = nn.Linear(768, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        out = self.mean_pooling(out[0], attention_mask)
        out = self.fc(out)
        return out