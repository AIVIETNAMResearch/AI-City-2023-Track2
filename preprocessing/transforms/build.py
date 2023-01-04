# encoding: utf-8

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
