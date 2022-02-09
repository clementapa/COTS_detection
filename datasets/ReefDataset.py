'''
    Author: Clément APAVOU
'''
import os, os.path as osp
import ast
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from .transforms import ToTensor

from utils.logger import init_logger

log_level = "DEBUG"
logger = init_logger("Dataloader", log_level)


class ReefDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 root_dir,
                 train,
                 augmentation=None,
                 transforms=None,
                 conv_bbox="pascal_voc"):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            train (boolean): Train set 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Ouput:
            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, 
                with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.conv_bbox = conv_bbox

        if augmentation is not None : self.augmentation = augmentation

        df = pd.read_csv(annotations_file)
        if train:
            df = df[(df.video_id == 0) | (df.video_id == 1)]
        else:
            df = df[(df.video_id == 2)]

        df = df[['video_id', 'video_frame', 'annotations']]
        self.img_annotations = df

        self.train = train

        if True:
            self.img_annotations = self.img_annotations[
                self.img_annotations["annotations"] != "[]"]

    def __getitem__(self, idx):

        path_base = osp.join(self.root_dir, "video_{}/{}.jpg")
        video_id = self.img_annotations.iloc[idx, 0]
        frame_id = self.img_annotations.iloc[idx, 1]
        path_img = path_base.format(video_id, frame_id)

        try:
            img = Image.open(path_img).convert("RGB")
            img.verify()  # Verify it is in fact an image
        except (IOError, SyntaxError) as e:
            # logger.warning('Bad file:', path_img)
            print(('Bad file:', path_img))

        boxes = []
        bounding_boxes = ast.literal_eval(self.img_annotations.iloc[idx, 2])
        if self.conv_bbox != "coco":
            for box in bounding_boxes:
                if self.conv_bbox == "pascal_voc":
                    boxes.append([
                        box['x'], box['y'], box['x'] + box['width'],
                        box['y'] + box['height']
                    ])
                elif self.conv_bbox == "yolo":  # implemente, choose dataset dans le fichier config
                    boxes.append(
                        [box['x'], box['y'], box['width'], box['height']])
                else:
                    pass

        if boxes == []:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((1, 1), dtype=torch.int64)
        else:
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((boxes.shape[0], ), dtype=torch.int64)

        image_id = f"{video_id}-{frame_id}"
        if self.conv_bbox == "pascal_voc":
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        elif self.conv_bbox == "yolo":
            area = boxes[:, 2] * boxes[:, 3] # FIXME to modify

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # transformations using albumentation library
        if self.transforms is not None:

            if len(target['boxes']) != 0:
                transformed = self.transforms(image=np.array(img),
                                              bboxes=target['boxes'],
                                              class_labels=target['labels'])
                img = torch.as_tensor(transformed['image'])
                target["boxes"] = torch.as_tensor(transformed['bboxes'],
                                                  dtype=torch.float32)
                target["labels"] = torch.as_tensor(transformed['class_labels'])
            else:  # negative samples
                if self.train:
                    transforms = []
                    transforms.append(A.Resize(width=self.augmentation.size.w, height=self.augmentation.size.h))
                    transforms.append(A.HorizontalFlip(p=0.5))
                    transforms.append(A.VerticalFlip(p=0.5))
                    transforms.append(A.RandomBrightnessContrast(p=0.6))
                    transforms.append(ToTensor())
                    transforms = A.Compose(transforms)
                    transformed = transforms(image=np.array(img))
                    img = torch.as_tensor(transformed['image'])
                else:
                    transforms = []
                    transforms.append(A.Resize(width=self.augmentation.size.w, height=self.augmentation.size.h))
                    transforms.append(ToTensor())
                    transforms = A.Compose(transforms)
                    transformed = transforms(image=np.array(img))
                    img = torch.as_tensor(transformed['image'])

        return img, target

    def __len__(self):
        return len(self.img_annotations)


def collate_fn(batch):
    return tuple(zip(*batch))
