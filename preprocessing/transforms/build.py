# encoding: utf-8
from googletrans import Translator
from transformers import MarianMTModel, MarianTokenizer

import torchvision
import torchvision.transforms as T

from .transforms import RandomErasing, RandomPatch, RandomResolution,ColorSpaceConvert, ColorAugmentation, RandomBlur, GaussianBlur
from .augmix import AugMix


def build_vanilla_transforms(cfg, is_train=True):
    if is_train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)],p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transform


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform = T.Compose([
            # RandomResolution(probability=0.5),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
            #T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
            #T.RandomApply([T.transforms.RandomAffine(degrees=20, scale=(1.0, 1.0))], p=cfg.INPUT.RANDOM_AFFINE_PROB),
            #T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            #GaussianBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),
            AugMix(prob=cfg.INPUT.AUGMIX_PROB),
            RandomBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),
            T.ToTensor(),
            normalize_transform,
            #ColorAugmentation(),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_motion_transform(cfg, vanilla=False):
    if vanilla:
        return build_vanilla_transforms(cfg, True)
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = T.Compose([
        # RandomResolution(probability=0.5),
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        # T.Pad(cfg.INPUT.PADDING),
        # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        # RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
        # T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
        # T.RandomApply([T.transforms.RandomAffine(degrees=20, scale=(1.0, 1.0))], p=cfg.INPUT.RANDOM_AFFINE_PROB),
        # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        # GaussianBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),
        # AugMix(prob=cfg.INPUT.AUGMIX_PROB),
        # RandomBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),
        T.ToTensor(),
        normalize_transform,
        # ColorAugmentation(),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    return transform



class BackTranslateAug(object):
    def __init__(self, first_model_name='Helsinki-NLP/opus-mt-en-vi', second_model_name='Helsinki-NLP/opus-mt-vi-en') -> None:
        self.first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model  = MarianMTModel.from_pretrained(first_model_name)

        self.second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model = MarianMTModel.from_pretrained(second_model_name)
    
    def format_batch_texts(self,language_code, batch_texts):
        formatted_batch = [">>{}<< {}".format(language_code, text) for text in batch_texts]

        return formatted_batch
    
    def translate(self, batch_texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        formated_batch_texts = self.format_batch_texts(language, batch_texts)
        
        # Generate translation using model
        translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

        # Convert the generated tokens indices back into text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        
        return translated_texts

    def __call__(self, original_texts, language="vi"):
        translated_text = self.translate(original_texts, self.first_model, self.first_model_tkn)
        back_translate_text = self.translate(translated_text, self.second_model, self.second_model_tkn)
        return back_translate_text
