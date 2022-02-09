'''
    Author: Clément APAVOU
'''
import random

from torchvision.transforms import functional as F
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform
from torchvision.transforms import ToTensor as transToTensor

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


class ToTensor(BasicTransform):
    def __init__(self, always_apply=True, p=1):
        super().__init__(always_apply=always_apply, p=p)
    
    @property
    def targets(self):
        return {'image': self.apply}
    
    def apply(self, img, **params):
        image = transToTensor()(img)
        return image

def get_transform(train, augmentation = None, format="pascal_voc"):
    ''' 
    format possible :
    "pascal_voc" [x_min, y_min, x_max, y_max], 
    "albumentations" normalized [x_min, y_min, x_max, y_max], 
    "coco" [x_min, y_min, width, height], 
    "yolo" normalized [x_center, y_center, width, height]
    '''
    transforms = []
    
    if train:
        # transforms.append(A.RandomSizedBBoxSafeCrop(width=augmentation.size.w, height=augmentation.size.h, erosion_rate=0.2))
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.VerticalFlip(p=0.5))
        transforms.append(A.RandomBrightnessContrast(p=0.6))
        # transforms.append(A.RandomFog(p=0.6, fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.08))
    else:
        transforms.append(A.Resize(width=augmentation.size.w, height=augmentation.size.h))

    transforms.append(ToTensor())
    
    return A.Compose(transforms, bbox_params=A.BboxParams(format=format, label_fields=['class_labels']))