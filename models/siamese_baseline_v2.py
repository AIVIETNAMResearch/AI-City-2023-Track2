import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet34
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from models.senet import se_resnext50_32x4d
from .efficientnet import EfficientNet
from .resnest import resnest50d
from loss import build_softmax_cls

# baseline for test

supported_img_encoders = ["se_resnext50_32x4d","efficientnet-b2","efficientnet-b3"]


class SiameseLocalandMotionModelBIG_Simple(torch.nn.Module):
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
                self.bnneck_car = nn.Sequential(nn.BatchNorm1d(self.img_in_dim),
                                                nn.Linear(self.img_in_dim, embed_dim))
                self.bnneck_motion = nn.Sequential(nn.BatchNorm1d(self.img_in_dim),
                                                nn.Linear(self.img_in_dim, embed_dim))
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
                self.bnneck_car = nn.Sequential(nn.BatchNorm1d(self.img_in_dim),
                                                nn.Linear(self.img_in_dim, embed_dim))
                self.bnneck_motion = nn.Sequential(nn.BatchNorm1d(self.img_in_dim),
                                                   nn.Linear(self.img_in_dim, embed_dim))

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        self.lnneck_text = nn.Sequential(nn.LayerNorm(embed_dim),
                                          nn.Linear(embed_dim, embed_dim),
                                          nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        for p in self.bert_model.parameters():
            p.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

        self.neck_merge = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim),
                                        nn.BatchNorm1d(embed_dim))

        if self.model_cfg.car_idloss:
            self.id_cls = nn.Linear(embed_dim, self.model_cfg.NUM_CLASS)
        if self.model_cfg.mo_idloss:   
            self.id_cls2 = nn.Linear(embed_dim, self.model_cfg.NUM_CLASS)
        if self.model_cfg.share_idloss:  
            self.id_cls3 = nn.Linear(embed_dim, self.model_cfg.NUM_CLASS)

    def forward(self, nl_input_ids, nl_attention_mask, crops, motion, targets=None):

        outputs = self.bert_model(nl_input_ids, attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds_ln = self.lnneck_text(lang_embeds)

        car_embeds = self.vis_backbone(crops)
        car_embeds = car_embeds.view(car_embeds.size(0), -1)
        car_embeds_bn = self.bnneck_car(car_embeds)

        motion_embeds = self.vis_backbone_bk(motion)
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        motion_embeds_bn = self.bnneck_motion(motion_embeds)

        # merge_embeds_bn = self.neck_merge(torch.cat([car_embeds, motion_embeds], dim=-1))
        merge_embeds_bn = self.neck_merge(torch.cat([car_embeds_bn, motion_embeds_bn], dim=-1))


        cls_logits_results = []
        if self.model_cfg.car_idloss:
            cls_logits = self.id_cls(car_embeds_bn)
            cls_logits_results.append(cls_logits)
        if self.model_cfg.mo_idloss:
            cls_logits2 = self.id_cls2(motion_embeds_bn)
            cls_logits_results.append(cls_logits2)

        if self.model_cfg.share_idloss:  
            merge_cls_t = self.id_cls3(lang_embeds_ln)
            merge_cls_v = self.id_cls3(merge_embeds_bn)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)

        merge_embeds_bn, lang_embeds_ln = map(lambda t: F.normalize(t, p=2, dim=-1), (merge_embeds_bn, lang_embeds_ln,))

        out = {
            "pairs": [(merge_embeds_bn, lang_embeds_ln), ],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out


class SiameseLocalandMotionModelBIG_V2(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        embed_dim = self.model_cfg.EMBED_DIM
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
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

        self.domian_vis_fc_merge = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
                                                 nn.Linear(embed_dim, embed_dim))
        self.vis_car_fc = nn.Sequential(nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2))
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2))
        self.vis_motion_fc = nn.Sequential(nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2))
        self.lang_motion_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim // 2))

        self.domian_lang_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                            nn.Linear(embed_dim, embed_dim))
        if self.model_cfg.car_idloss:
            pre_shared_cls1 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls1 = nn.Sequential(*pre_shared_cls1)
            self.id_cls1 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.CAR_CLS)
            # self.id_cls = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
            #                             nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.mo_idloss:
            pre_shared_cls2 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls2 = nn.Sequential(*pre_shared_cls2)
            self.id_cls2 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.MO_CLS)
            # self.id_cls2 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
            #                              nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.share_idloss:
            pre_shared_cls3 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls3 = nn.Sequential(*pre_shared_cls3)
            self.id_cls3 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)
            # self.id_cls3 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
            #                              nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))

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
        outputs = self.bert_model(nl_input_ids, attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)

        cls_logits_results = []
        if self.training and self.model_cfg.car_idloss:
            cls_logits1 = self.pre_id_cls1(visual_embeds)
            cls_logits1 = self.id_cls1(cls_logits1, targets=targets)
            cls_logits_results.append(cls_logits1)
        if self.training and self.model_cfg.mo_idloss:
            cls_logits2 = self.pre_id_cls2(motion_embeds)
            cls_logits2 = self.id_cls2(cls_logits2, targets=targets)
            cls_logits_results.append(cls_logits2)

        if self.training and self.model_cfg.share_idloss:
            merge_cls_t = self.pre_id_cls3(lang_embeds)
            merge_cls_t = self.id_cls3(merge_cls_t, targets=targets)
            merge_cls_v = self.pre_id_cls3(visual_merge_embeds)
            merge_cls_v = self.id_cls3(merge_cls_v, targets=targets)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)

        visual_merge_embeds, lang_merge_embeds, visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds = map(
            lambda t: F.normalize(t, p=2, dim=-1),
            (visual_merge_embeds, lang_embeds, visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds))

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out


class SiameseLocalandMotionModelBIG_DualTextCat(torch.nn.Module):
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
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

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

    def encode_text(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask,
                    clip_car_features_text=None):
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)
        if clip_car_features_text is not None:
            print(clip_car_features_text.shape, " LANG CAR")
            lang_car_embeds = torch.add(lang_car_embeds, clip_car_features_text)

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

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion, clip_car_features_vis=None):
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        if clip_car_features_vis is not None:
            print(clip_car_features_vis.shape, " LANG CAR")
            visual_car_embeds = torch.add(visual_car_embeds, clip_car_features_vis)

        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

            visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))
        else:
            visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, visual_mo_embeds))
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask,
                crops, motion, targets=None, clip_car_features_text=None, clip_car_features_vis=None):
        # text
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)
        
        if clip_car_features_text is not None:
            print(clip_car_features_text.shape, " LANG CAR")
            
            lang_car_embeds = torch.add(lang_car_embeds, clip_car_features_text)

        if clip_car_features_vis is not None:
            print(clip_car_features_vis.shape, " LANG CAR")
            visual_car_embeds = torch.add(visual_car_embeds, clip_car_features_vis)

        # visual
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        if self.model_cfg.HEAD.CAT_TRAIN:
            lang_merge_embeds = self.lang_fc_merge(torch.cat([lang_car_embeds, lang_motion_embeds], dim=-1))
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

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

        visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds = map(lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_motion_embeds))

        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds, lang_merge_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, lang_merge_embeds))
        else:
            lang_merge_embeds = torch.cat([lang_car_embeds, lang_mo_embeds], dim=-1)
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out


class SiameseLocalandMotionModelBIG_DualTextAdd(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        embed_dim = self.model_cfg.EMBED_DIM

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
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        # text model
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.lang_mo_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        if self.model_cfg.HEAD.ADD_TRAIN:
            self.vis_fc_merge = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim), nn.ReLU(),
                                              nn.Linear(embed_dim, embed_dim))
            self.lang_fc_merge = nn.Sequential(nn.LayerNorm(embed_dim),
                                               nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                               nn.Linear(embed_dim, embed_dim))

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
            pre_shared_cls3 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls3 = nn.Sequential(*pre_shared_cls3)
            self.id_cls3 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)

    def encode_text(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask):
        # text
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        if self.model_cfg.HEAD.ADD_TRAIN:
            lang_merge_embeds = (lang_car_embeds + lang_motion_embeds)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)
            lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_merge_embeds, lang_car_embeds,lang_motion_embeds))
        else:
            lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_car_embeds, lang_motion_embeds))
            lang_merge_embeds = (lang_car_embeds + lang_motion_embeds)

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion):
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        if self.model_cfg.HEAD.ADD_TRAIN:
            visual_merge_embeds = (visual_car_embeds + visual_mo_embeds)
            visual_merge_embeds = self.vis_fc_merge(visual_merge_embeds)
            visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))
        else:
            visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, visual_mo_embeds))
            visual_merge_embeds = (visual_car_embeds + visual_mo_embeds)

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask, crops, motion, targets=None):
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

        if self.model_cfg.HEAD.ADD_TRAIN:
            lang_merge_embeds = (lang_car_embeds + lang_motion_embeds)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

            visual_merge_embeds = (visual_car_embeds + visual_mo_embeds)
            visual_merge_embeds = self.vis_fc_merge(visual_merge_embeds)

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

        if self.training and self.model_cfg.share_idloss and self.model_cfg.HEAD.ADD_TRAIN:
            merge_cls_t = self.pre_id_cls3(lang_merge_embeds)
            merge_cls_t = self.id_cls3(merge_cls_t, targets=targets)
            cls_logits_results.append(merge_cls_t)

            merge_cls_v = self.pre_id_cls3(visual_merge_embeds)
            merge_cls_v = self.id_cls3(merge_cls_v, targets=targets)
            cls_logits_results.append(merge_cls_v)

        if self.model_cfg.HEAD.ADD_TRAIN:
            visual_merge_embeds, lang_merge_embeds, visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, lang_merge_embeds, visual_car_embeds, lang_car_embeds, visual_mo_embeds,
                 lang_motion_embeds))
        else:
            visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_motion_embeds))
            lang_merge_embeds = (lang_car_embeds + lang_motion_embeds)
            visual_merge_embeds = (visual_car_embeds + visual_mo_embeds)

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out