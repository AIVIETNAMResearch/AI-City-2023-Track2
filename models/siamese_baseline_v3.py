import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet34
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel, DebertaTokenizer, DebertaModel
from models.senet import se_resnext50_32x4d
from .efficientnet import EfficientNet
import timm
from .aggregation_head import ContextualizedWeightedHead
from .resnest import resnest50d
from loss import build_softmax_cls

supported_img_encoders = ["se_resnext50_32x4d","efficientnet-b2","efficientnet-b3", "ViT"]

# Define Transformer architecture
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, nhead=8, dim_feedforward=hidden_size), num_layers=num_layers)
        self.output = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.transformer(x)
        x = x[-1] # take the last output vector
        x = self.output(x)
        return x
    

class SiameseLocalandMotionModelBIGV3(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        embed_dim = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "ViT":
                self.vis_backbone = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
                self.vis_backbone_bk = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)

                self.img_in_dim = 1000
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)


            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"
        
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        bert_out_dim = 1024
        for p in  self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        
        self.domian_vis_fc_merge = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim),nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.vis_car_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.vis_motion_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_motion_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))

        self.domian_lang_fc = nn.Sequential(nn.LayerNorm(bert_out_dim),nn.Linear(bert_out_dim, bert_out_dim), nn.ReLU(), nn.Linear(bert_out_dim, embed_dim))
        if self.model_cfg.car_idloss:
            self.id_cls = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.mo_idloss:   
            self.id_cls2 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.share_idloss:  
            self.id_cls3 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))

    def encode_text(self, nl_input_ids, nl_attention_mask):
        outputs = self.bert_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        # lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)

        lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (lang_embeds, lang_car_embeds, lang_mo_embeds))

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion):
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))
        # visual_embeds = F.normalize(visual_merge_embeds, p=2, dim=-1)

        visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_input_ids, nl_attention_mask, crops, motion, targets=None):

        outputs = self.bert_model(nl_input_ids,attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)        
        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds,visual_mo_embeds],dim=-1))
        cls_logits_results = []
        if self.model_cfg.car_idloss:
            cls_logits = self.id_cls(visual_embeds)
            cls_logits_results.append(cls_logits)
        if self.model_cfg.mo_idloss:
            cls_logits2 = self.id_cls2(motion_embeds)
            cls_logits_results.append(cls_logits2)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        if self.model_cfg.share_idloss:  
            merge_cls_t = self.id_cls3(lang_embeds)
            merge_cls_v = self.id_cls3(visual_merge_embeds)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)

        visual_merge_embeds, lang_merge_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds = map(lambda t: F.normalize(t, p = 2, dim = -1), (visual_merge_embeds, lang_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds))

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                      (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out


class SiameseLocalandMotionModelMultiFrames(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        embed_dim = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "ViT":
                self.vis_backbone = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True)
                self.vis_backbone_bk = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True)
                self.vis_backbone_context = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True)

                self.img_in_dim = 1000
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_context = nn.Linear(self.img_in_dim, embed_dim)


            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.vis_backbone_context = se_resnext50_32x4d()

                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
                self.domian_vis_fc_context = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)

            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_context = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_context_transformer = TransformerEncoder(input_size=embed_dim, hidden_size=512, num_layers=2, output_size=embed_dim)

                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_context = nn.Linear(self.img_in_dim, embed_dim)

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"
        
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        bert_out_dim = 1024
        for p in  self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        
        self.domian_vis_fc_merge = nn.Sequential(nn.Linear(embed_dim * 1.5, embed_dim), nn.BatchNorm1d(embed_dim),nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.vis_car_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.vis_motion_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_motion_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.vis_context_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))

        self.domian_lang_fc = nn.Sequential(nn.LayerNorm(bert_out_dim),nn.Linear(bert_out_dim, bert_out_dim), nn.ReLU(), nn.Linear(bert_out_dim, embed_dim))
        if self.model_cfg.car_idloss:
            self.id_cls = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.mo_idloss:   
            self.id_cls2 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.share_idloss:  
            self.id_cls3 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.con_idloss:
            self.id_cls4 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))

    def encode_text(self, nl_input_ids, nl_attention_mask):
        outputs = self.bert_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        # lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)

        lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (lang_embeds, lang_car_embeds, lang_mo_embeds))

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion, frames):
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)

        b, s, c, h, w = frames.size()
        frames = frames.view(b*s, c, h, w)

        context_embeds = self.domian_vis_fc_context(self.vis_backbone_context(frames))
        context_embeds = context_embeds.view(b, s, -1)
        context_embeds = context_embeds.permute(1, 0, 2)
        context_embeds = self.vis_backbone_context(context_embeds)

        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_context_embeds = self.vis_context_fc(context_embeds)

        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds, visual_context_embeds], dim=-1))

        # visual_embeds = F.normalize(visual_merge_embeds, p=2, dim=-1)

        visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_context_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_context_embeds))

        return [visual_car_embeds, visual_mo_embeds, visual_context_embeds, visual_merge_embeds]

    def forward(self, nl_input_ids, nl_attention_mask, crops, motion, frames, targets=None):

        outputs = self.bert_model(nl_input_ids,attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)     

        b, s, c, h, w = frames.size()
        frames = frames.view(b*s, c, h, w)

        context_embeds = self.domian_vis_fc_context(self.vis_backbone_context(frames))
        context_embeds = context_embeds.view(b, s, -1)
        context_embeds = context_embeds.permute(1, 0, 2)
        context_embeds = self.vis_backbone_context(context_embeds)

        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_context_embeds = self.vis_context_fc(context_embeds)

        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds,visual_mo_embeds, visual_merge_embeds],dim=-1))
        cls_logits_results = []
        if self.model_cfg.car_idloss:
            cls_logits = self.id_cls(visual_embeds)
            cls_logits_results.append(cls_logits)
        if self.model_cfg.mo_idloss:
            cls_logits2 = self.id_cls2(motion_embeds)
            cls_logits_results.append(cls_logits2)
        
        if self.model_cfg.con_idloss:
            cls_logits4 = self.id_cls4(context_embeds)
            cls_logits_results.append(cls_logits4)

        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        if self.model_cfg.share_idloss:  
            merge_cls_t = self.id_cls3(lang_embeds)
            merge_cls_v = self.id_cls3(visual_merge_embeds)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)

        visual_merge_embeds, lang_merge_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds, visual_context_embeds = map(lambda t: F.normalize(t, p = 2, dim = -1), (visual_merge_embeds, lang_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds, visual_context_embeds))

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds), 
                      (visual_context_embeds, lang_mo_embeds), (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out
    
class SiameseLocalandMotionModelBIG_DualTextCat_Multiframes(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        embed_dim = self.model_cfg.EMBED_DIM
        double_embed_dim = 2 * embed_dim
        merge_dim = self.model_cfg.MERGE_DIM

        # visual model
        if self.model_cfg.IMG_ENCODER in supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
            elif self.model_cfg.IMG_ENCODER == "resnest50":
                self.vis_backbone = resnest50d()
                self.vis_backbone_bk = resnest50d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_context = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_context_transformer = TransformerEncoder(input_size=embed_dim, hidden_size=512, num_layers=2, output_size=embed_dim)

                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_context = nn.Linear(self.img_in_dim, embed_dim)
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        # text model
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.lang_mo_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        if self.model_cfg.HEAD.CAT_TRAIN:
            self.vis_fc_merge = nn.Sequential(nn.Linear(3 * embed_dim, double_embed_dim),
                                              nn.BatchNorm1d(double_embed_dim), nn.ReLU(),
                                              nn.Linear(double_embed_dim, merge_dim))
            self.lang_fc_merge = nn.Sequential(nn.LayerNorm(double_embed_dim),
                                               nn.Linear(double_embed_dim, double_embed_dim), nn.ReLU(),
                                               nn.Linear(double_embed_dim, merge_dim))

        # cls model
        if self.model_cfg.car_idloss:
            pre_shared_cls1 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls1 = nn.Sequential(*pre_shared_cls1)
            self.id_cls1 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.CAR_CLS)

        if self.model_cfg.mo_idloss:
            pre_shared_cls2 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls2 = nn.Sequential(*pre_shared_cls2)
            self.id_cls2 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.MO_CLS)

        if self.model_cfg.share_idloss:
            pre_shared_cls3 = [nn.Linear(merge_dim, merge_dim), nn.BatchNorm1d(merge_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls3 = nn.Sequential(*pre_shared_cls3)
            self.id_cls3 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)

        if self.model_cfg.con_idloss:
            pre_shared_cls4 = [nn.Linear(merge_dim, merge_dim), nn.BatchNorm1d(merge_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls4 = nn.Sequential(*pre_shared_cls4)
            self.id_cls4 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)

    def encode_text(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask):
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        if self.model_cfg.HEAD.CAT_TRAIN:
            if lang_motion_embeds.shape[0] != lang_car_embeds.shape[0]:
                lang_merge_embeds = torch.cat([lang_car_embeds.repeat(lang_motion_embeds.shape[0], 1), lang_motion_embeds], dim=-1)
            else:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds, lang_motion_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

            lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_merge_embeds, lang_car_embeds, lang_motion_embeds))
        else:
            lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_car_embeds, lang_motion_embeds))

            if lang_motion_embeds.shape[0] != lang_car_embeds.shape[0]:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds.repeat(lang_motion_embeds.shape[0], 1), lang_motion_embeds], dim=-1)
            else:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds, lang_motion_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

        return [lang_car_embeds, lang_mo_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion, frames):
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        b, s, c, h, w = frames.size()
        frames = frames.view(b*s, c, h, w)

        visual_con_embeds = self.domian_vis_fc_context(self.vis_backbone_context(frames))
        visual_con_embeds = visual_con_embeds.view(b, s, -1)
        visual_con_embeds = visual_con_embeds.permute(1, 0, 2)
        visual_con_embeds = self.vis_context_transformer(visual_con_embeds)


        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds, visual_con_embeds], dim=-1))

            visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_con_embeds  = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_con_embeds))
        else:
            visual_car_embeds, visual_mo_embeds,visual_con_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, visual_mo_embeds, visual_con_embeds))
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds, visual_con_embeds], dim=-1)

        return [visual_car_embeds, visual_mo_embeds, visual_con_embeds, visual_merge_embeds]

    def forward(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask,
                crops, motion, frames, targets=None):
        # text
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        # visual
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        b, s, c, h, w = frames.size()
        frames = frames.view(b*s, c, h, w)

        visual_con_embeds = self.domian_vis_fc_context(self.vis_backbone_context(frames))
        visual_con_embeds = visual_con_embeds.view(b, s, -1)
        visual_con_embeds = visual_con_embeds.permute(1, 0, 2)
        visual_con_embeds = self.vis_context_transformer(visual_con_embeds)

        if self.model_cfg.HEAD.CAT_TRAIN:
            lang_merge_embeds = self.lang_fc_merge(torch.cat([lang_car_embeds, lang_motion_embeds], dim=-1))
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds, visual_con_embeds], dim=-1))

        # cls
        cls_logits_results = []
        if self.training and self.model_cfg.car_idloss:
            car_cls_t = self.pre_id_cls1(lang_car_embeds)
            car_cls_t = self.id_cls1(car_cls_t, targets=targets)
            cls_logits_results.append(car_cls_t)

            car_cls_v = self.pre_id_cls1(visual_car_embeds)
            car_cls_v = self.id_cls1(car_cls_v, targets=targets)
            cls_logits_results.append(car_cls_v)

        if self.training and self.model_cfg.mo_idloss:
            motion_cls_t = self.pre_id_cls2(lang_motion_embeds)
            motion_cls_t = self.id_cls2(motion_cls_t, targets=targets)
            cls_logits_results.append(motion_cls_t)

            motion_cls_v = self.pre_id_cls2(visual_mo_embeds)
            motion_cls_v = self.id_cls2(motion_cls_v, targets=targets)
            cls_logits_results.append(motion_cls_v)

        if self.training and self.model_cfg.share_idloss and self.model_cfg.HEAD.CAT_TRAIN:
            merge_cls_t = self.pre_id_cls3(lang_merge_embeds)
            merge_cls_t = self.id_cls3(merge_cls_t, targets=targets)
            cls_logits_results.append(merge_cls_t)

            merge_cls_v = self.pre_id_cls3(visual_merge_embeds)
            merge_cls_v = self.id_cls3(merge_cls_v, targets=targets)
            cls_logits_results.append(merge_cls_v)

        if self.training and self.model_cfg.con_idloss:
            context_cls_t = self.pre_id_cls2(lang_motion_embeds)
            context_cls_t = self.id_cls2(context_cls_t, targets=targets)
            cls_logits_results.append(context_cls_t)

            context_cls_v = self.pre_id_cls2(visual_con_embeds)
            context_cls_v = self.id_cls2(context_cls_v, targets=targets)
            cls_logits_results.append(context_cls_v)


        visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds, visual_con_embeds = map(lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_motion_embeds, visual_con_embeds))

        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds, lang_merge_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, lang_merge_embeds))
        else:
            lang_merge_embeds = torch.cat([lang_car_embeds, lang_mo_embeds], dim=-1)
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds, visual_con_embeds], dim=-1)

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds), (visual_con_embeds, lang_mo_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out
    

class SiameseLocalandMotionModelBIG_DualTextCat_view_MultiCrops(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        self.logit_scale_nl = nn.Parameter(torch.ones(()), requires_grad=True)
        embed_dim = self.model_cfg.EMBED_DIM
        double_embed_dim = 2 * embed_dim
        merge_dim = self.model_cfg.MERGE_DIM

        # visual model
        if self.model_cfg.IMG_ENCODER in supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "ViT":
                self.vis_backbone = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
                self.vis_backbone_bk = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)

                self.img_in_dim = 1000
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
            
            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
            elif self.model_cfg.IMG_ENCODER == "resnest50":
                self.vis_backbone = resnest50d()
                self.vis_backbone_bk = resnest50d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.img_in_dim = self.vis_backbone.out_channels

                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        self.transformer_merge = TransformerEncoder(embed_dim, hidden_size=512, num_layers=2, 
                                                    output_size=embed_dim)
        # text model
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.lang_mo_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        if self.model_cfg.HEAD.CAT_TRAIN:
            self.vis_fc_merge = nn.Sequential(nn.Linear(double_embed_dim, double_embed_dim),
                                              nn.BatchNorm1d(double_embed_dim), nn.ReLU(),
                                              nn.Linear(double_embed_dim, merge_dim))
            self.lang_fc_merge = nn.Sequential(nn.LayerNorm(double_embed_dim),
                                               nn.Linear(double_embed_dim, double_embed_dim), nn.ReLU(),
                                               nn.Linear(double_embed_dim, merge_dim))

        # cls model
        if self.model_cfg.car_idloss:
            pre_shared_cls1 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls1 = nn.Sequential(*pre_shared_cls1)
            self.id_cls1 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.CAR_CLS)

        if self.model_cfg.mo_idloss:
            pre_shared_cls2 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls2 = nn.Sequential(*pre_shared_cls2)
            self.id_cls2 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.MO_CLS)

        if self.model_cfg.share_idloss:
            pre_shared_cls3 = [nn.Linear(merge_dim, merge_dim), nn.BatchNorm1d(merge_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls3 = nn.Sequential(*pre_shared_cls3)
            self.id_cls3 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)

    def encode_text(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask):
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        if lang_motion_embeds.shape[0] != lang_car_embeds.shape[0]:
            lang_merge_embeds = torch.cat([lang_car_embeds.repeat(lang_motion_embeds.shape[0], 1), lang_motion_embeds],
                                          dim=-1)
        else:
            lang_merge_embeds = torch.cat(
                [lang_car_embeds, lang_motion_embeds], dim=-1)
        lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

        lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (lang_merge_embeds, lang_car_embeds, lang_motion_embeds))

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion):
        batch_size, seq_len, channels, height, width = crops.size()
        crops = crops.view(batch_size*seq_len, channels, height, width)

        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(batch_size, seq_len, -1)
        visual_car_embeds = visual_car_embeds.permute(1, 0, 2)
        visual_car_embeds = self.transformer_merge(visual_car_embeds)


        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

        visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask,
                nl_view_input_ids, nl_view_attention_mask, crops, motion, targets=None):
        # text
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        lang_merge_embeds = self.lang_fc_merge(torch.cat([lang_car_embeds, lang_motion_embeds], dim=-1))

        # visual
        batch_size, seq_len, channels, height, width = crops.size()
        crops = crops.view(batch_size*seq_len, channels, height, width)
        
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(batch_size, seq_len, -1)
        visual_car_embeds = visual_car_embeds.permute(1, 0, 2)
        visual_car_embeds = self.transformer_merge(visual_car_embeds)

        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)


        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

        # view nlp
        if self.training:
            outputs_view = self.bert_model(nl_view_input_ids, attention_mask=nl_view_attention_mask)
            lang_embeds_view = torch.mean(outputs_view.last_hidden_state, dim=1)
            lang_embeds_view = self.lang_mo_fc(lang_embeds_view)
            lang_embeds_view = F.normalize(lang_embeds_view, p=2, dim=-1)
        else:
            lang_embeds_view = torch.tensor(0., device=visual_merge_embeds.device)

        # cls
        cls_logits_results = []
        if self.training and self.model_cfg.car_idloss:
            car_cls_t = self.pre_id_cls1(lang_car_embeds)
            car_cls_t = self.id_cls1(car_cls_t, targets=targets)
            cls_logits_results.append(car_cls_t)

            car_cls_v = self.pre_id_cls1(visual_car_embeds)
            car_cls_v = self.id_cls1(car_cls_v, targets=targets)
            cls_logits_results.append(car_cls_v)

        if self.training and self.model_cfg.mo_idloss:
            motion_cls_t = self.pre_id_cls2(lang_motion_embeds)
            motion_cls_t = self.id_cls2(motion_cls_t, targets=targets)
            cls_logits_results.append(motion_cls_t)

            motion_cls_v = self.pre_id_cls2(visual_mo_embeds)
            motion_cls_v = self.id_cls2(motion_cls_v, targets=targets)
            cls_logits_results.append(motion_cls_v)

        if self.training and self.model_cfg.share_idloss:
            merge_cls_t = self.pre_id_cls3(lang_merge_embeds)
            merge_cls_t = self.id_cls3(merge_cls_t, targets=targets)
            cls_logits_results.append(merge_cls_t)

            merge_cls_v = self.pre_id_cls3(visual_merge_embeds)
            merge_cls_v = self.id_cls3(merge_cls_v, targets=targets)
            cls_logits_results.append(merge_cls_v)

        visual_merge_embeds, lang_merge_embeds, visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds, \
            = map(lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, lang_merge_embeds,
             visual_car_embeds, lang_car_embeds,
             visual_mo_embeds, lang_motion_embeds))

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
            "view_nl": (lang_embeds_view, self.logit_scale_nl)
        }

        return out
    
class SiameseLocalandMotionModelBIG_MultiQueries(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        embed_dim = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        bert_out_dim = 1024
        for p in  self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        
        self.domian_vis_fc_merge = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim),nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.vis_car_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.vis_motion_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_motion_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))

        self.domian_lang_fc = nn.Sequential(nn.LayerNorm(bert_out_dim),nn.Linear(bert_out_dim, bert_out_dim), nn.ReLU(), nn.Linear(bert_out_dim, embed_dim))
        if self.model_cfg.car_idloss:
            self.id_cls = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.mo_idloss:   
            self.id_cls2 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.share_idloss:  
            self.id_cls3 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))

        self.clip_feats_fc_text = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.LayerNorm(embed_dim), nn.ReLU(),nn.Linear(embed_dim, embed_dim))
        self.clip_feats_fc_vis = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.LayerNorm(embed_dim), nn.ReLU(),nn.Linear(embed_dim, embed_dim))

        self.text_head = ContextualizedWeightedHead(d_model= embed_dim, nhead = 8, num_layers=2, fc_dim_list= [embed_dim, embed_dim*2, 1])

    def encode_text(self, nl_input_ids_all, nl_attention_mask_all):
        lang_embeds = []

        for nl_input_ids, nl_attention_mask in zip(nl_input_ids_all, nl_attention_mask_all):
            outputs = self.bert_model(nl_input_ids,
                                    attention_mask=nl_attention_mask)
            lang_embs = torch.mean(outputs.last_hidden_state, dim=1)
            lang_embs = self.domian_lang_fc(lang_embs)

            lang_embeds.append(lang_embs)
            # lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)

        lang_embeds = torch.stack(lang_embeds)

        lang_embeds = self.text_head(lang_embeds)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)


        lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (lang_embeds, lang_car_embeds, lang_mo_embeds))

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion, clip_feats_vis=None):
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        if clip_feats_vis is not None:
            #print(torch.squeeze(clip_feats_vis).shape)
            clip_feats_vis = self.clip_feats_fc_vis(torch.squeeze(clip_feats_vis))
            visual_embeds = torch.add(visual_embeds, clip_feats_vis)

        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        visual_car_embeds = self.vis_car_fc(visual_embeds)

        
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))
        # visual_embeds = F.normalize(visual_merge_embeds, p=2, dim=-1)

        visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_input_ids_all, nl_attention_mask_all, crops, motion, targets=None):

        lang_embeds = []

        for nl_input_ids, nl_attention_mask in zip(nl_input_ids_all, nl_attention_mask_all):
            outputs = self.bert_model(nl_input_ids,
                                    attention_mask=nl_attention_mask)
            lang_embs = torch.mean(outputs.last_hidden_state, dim=1)
            lang_embs = self.domian_lang_fc(lang_embs)

            lang_embeds.append(lang_embs)
            # lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)

        lang_embeds = torch.stack(lang_embeds)

        lang_embeds = self.text_head(lang_embeds)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)


        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)


        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)        
        visual_car_embeds = self.vis_car_fc(visual_embeds)



        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds,visual_mo_embeds],dim=-1))
        cls_logits_results = []
        if self.model_cfg.car_idloss:
            cls_logits = self.id_cls(visual_embeds)
            cls_logits_results.append(cls_logits)
        if self.model_cfg.mo_idloss:
            cls_logits2 = self.id_cls2(motion_embeds)
            cls_logits_results.append(cls_logits2)


        if self.model_cfg.share_idloss:  
            merge_cls_t = self.id_cls3(lang_embeds)
            merge_cls_v = self.id_cls3(visual_merge_embeds)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)

        visual_merge_embeds, lang_merge_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds = map(lambda t: F.normalize(t, p = 2, dim = -1), (visual_merge_embeds, lang_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds))

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                      (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out