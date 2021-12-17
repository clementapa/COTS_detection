import os, os.path as osp 
import ast
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import datasets.transforms as T
from utils.logger import init_logger
log_level = "DEBUG"
logger = init_logger("Dataloader", log_level)

class ReefDataset(Dataset):
    def __init__(self, annotations_file, root_dir, train, transform=None):
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
        self.transform = transform

        df = pd.read_csv(annotations_file)
        if train:
            df = df[(df.video_id == 0) | (df.video_id == 1)]
        else: 
            df = df[(df.video_id == 2)]
            
        df = df[['video_id', 'video_frame', 'annotations']]
        self.img_annotations = df

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx): 
        
        path_base = osp.join(self.root_dir, "video_{}/{}.jpg")
        video_id = self.img_annotations.iloc[idx, 0]
        frame_id = self.img_annotations.iloc[idx, 1]
        path_img = path_base.format(video_id, frame_id)

        try:
            img = Image.open(path_img).convert("RGB")
            img.verify() # Verify it is in fact an image
        except (IOError, SyntaxError) as e:
            # logger.warning('Bad file:', path_img)
            print(('Bad file:', path_img))
        
        boxes = []
        bounding_boxes = ast.literal_eval(self.img_annotations.iloc[idx, 2])
        for box in bounding_boxes:
            boxes.append([box['x'], box['y'], box['x']+box['width'], box['y']+box['height']])
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones(((boxes.shape[0]),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = f"{video_id}-{frame_id}"
        # target["path_img"] = path_img

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_boxes(img, bounding_boxes):
    img = img.permute(1, 2, 0).numpy().copy()
    bounding_boxes = bounding_boxes.numpy().astype(np.int32)
    for box in bounding_boxes:
        # draw = ImageDraw.Draw(img)
        # x0, y0, x1, y1 = box
        # draw.rectangle((x0, y0, x1, y1), outline=180, width=5)
        cv2.rectangle(img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (220, 0, 0), 3)
    return img

# csv_file = "/home/clement/Documents/Cours/MVA/Cours_to_validate/Deep_learning/Project/tensorflow-great-barrier-reef/train.csv"
# root_dir = "/home/clement/Documents/Cours/MVA/Cours_to_validate/Deep_learning/Project/tensorflow-great-barrier-reef/train_images"
# train_data = ReefDataset(csv_file, root_dir, train=True, transform=get_transform(True))
# valid_data = ReefDataset(csv_file, root_dir, train=False, transform=get_transform(False))

# train_loader = DataLoader(train_data, batch_size=20, shuffle=False, collate_fn=collate_fn)
# valid_loader = DataLoader(valid_data, batch_size=20, shuffle=False, collate_fn=collate_fn)

# images, targets = next(iter(train_loader))

# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]

# for img, target in zip(images, targets):
#     print(target["path_img"])
#     if target["boxes"].shape[0] != 0:
#         img = draw_boxes(img, target["boxes"])
#         plt.imshow(img)
#         plt.show()

# breakpoint()