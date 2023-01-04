# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
import numpy as np
import cv2
from collections import deque
from PIL import Image, ImageFilter
import torch

class RandomResolution(object):

    def __init__(self, probability=0.33):
        self.probability = probability

    def __call__(self, img):
        h, w = img.size
        if h < 100 and w < 100:
            return img
        if random.uniform(0, 1) >= self.probability:
            return img

        s = max(h, w)
        _50_s, _100_s = s / 50, s / 100
        scale = random.uniform(_50_s, _100_s)
        new_h, new_w = int(h / scale), int(w / scale)
        img = img.resize((new_h, new_w))
        return img

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ColorSpaceConvert(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, color_space='rgb', phase='train'):
        self.color_space = color_space
        self.phase = phase

    def __call__(self, img):
        if self.color_space == 'rgb':
            img = img
        elif self.color_space == 'define':
            img = np.asarray(img).astype(np.uint8)
            new_img = cv2.merge([255 - img[:, :, 0], 255 - img[:, :, 1], img[:, :, 2]]).astype(np.uint8)
            img = Image.fromarray(new_img)
        return img


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.

    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
            self,
            prob_happen=0.5,
            pool_capacity=50000,
            min_sample_size=100,
            patch_min_area=0.01,
            patch_max_area=0.5,
            patch_min_ratio=0.1,
            prob_rotate=0.5,
            prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area
            ) * area
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_ratio
            )
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img

class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.
    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.
    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, p=0.5, sigma=[.1, 1.]):
        self.p = p
        self.sigma = sigma

    def __call__(self, image):
        if random.uniform(0, 1) > self.p:
            return image
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image



class RandomBlur(object):
    def __init__(self, p=0.5, sigma=1.3):
        self.p = p
        self.sigma = sigma

    def __call__(self, image): # image : numpy , [height, width, channel]
        if random.uniform(0, 1) > self.p:
            return image
        return cv2.GaussianBlur(image, (5, 5), self.sigma)
