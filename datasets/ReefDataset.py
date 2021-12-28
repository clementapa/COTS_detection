import os, os.path as osp
import ast
from PIL import Image
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np

from utils.logger import init_logger

log_level = "DEBUG"
logger = init_logger("Dataloader", log_level)


class ReefDataset(Dataset):
    def __init__(self, annotations_file, root_dir, train, transforms=None):
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

        df = pd.read_csv(annotations_file)
        if train:
            df = df[(df.video_id == 0) | (df.video_id == 1)]
        else:
            df = df[(df.video_id == 2)]

        df = df[['video_id', 'video_frame', 'annotations']]
        self.img_annotations = df

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
        for box in bounding_boxes:
            boxes.append([
                box['x'], box['y'], box['x'] + box['width'],
                box['y'] + box['height']
            ])

        if boxes == []:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((1, 1), dtype=torch.int64)
        else:
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((boxes.shape[0], ), dtype=torch.int64)

        image_id = f"{video_id}-{frame_id}"
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        # transformations using albumentation library
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img),
                                          bboxes=target['boxes'],
                                          class_labels=target['labels'])
            img = torch.as_tensor(transformed['image'], dtype=torch.float32)
            target["boxes"] = torch.as_tensor(transformed['bboxes'])
            target["labels"] = torch.as_tensor(transformed['class_labels'])

        return img, target

    def __len__(self):
        return len(self.img_annotations)


def collate_fn(batch):
    return tuple(zip(*batch))
