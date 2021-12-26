import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import albumentations as A


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]

            if bbox.shape[0] != 0:
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform_albu(train):
        
    transforms = []
    transforms.append(A.RandomSizedBBoxSafeCrop(width=640, height=360, erosion_rate=0.2)) # FIXME taille image entrainement doit etre égale à la taille des images de validation?
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.RandomBrightnessContrast(p=0.6))

    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def get_transform(train):

    transforms = []
    transforms.append(ToTensor())
    if not train: return Compose(transforms)

    return Compose(transforms)