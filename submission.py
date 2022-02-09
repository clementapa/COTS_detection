'''
    Author: Clément APAVOU
'''
import sys

import argparse
from albumentations.augmentations import transforms

import numpy as np

import torch

from logger import init_logger
import utils as utils
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import albumentations as A
from utils import draw_predictions_and_targets
import matplotlib.pyplot as plt

def format_prediction_string(boxes, scores):
    # Format as specified in the evaluation page
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.2f} {1} {2} {3} {4}".format(
            j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def predict(model, pixel_array, detection_threshold, device, resize=None, verbose=False):
    # Predictions for a single image

    # Apply all the transformations that are required
    pixel_array = pixel_array.astype(np.float32) / 255.
    if resize != None:
        pixel_array = A.Resize(width=resize[0], height=resize[1])(image=pixel_array)['image']
    tensor_img = ToTensorV2(p=1.0)(image=pixel_array)['image'].unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        outputs = model(tensor_img.to(device))[0]

    # Move predictions to cpu and numpy
    boxes = outputs['boxes'].data.cpu().numpy()
    scores = outputs['scores'].data.cpu().numpy()

    # Filter predictions with low score
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    scores = scores[scores >= detection_threshold]

    if verbose:
        dict_draw = draw_predictions_and_targets(pixel_array, boxes)
        plt.imshow(dict_draw["img_pred"])
        plt.savefig("resize")

    if resize != None and len(boxes) != 0:
        transforms = [A.Resize(width=1280, height=720)]  # original size
        transform = A.Compose(transforms,
                              bbox_params=A.BboxParams(
                                  format='pascal_voc',
                                  label_fields=['class_labels']))
        
        transformed = transform(image=pixel_array, bboxes=boxes, class_labels=torch.ones((boxes.shape[0], 1), dtype=torch.int64))
        img_originale = transformed['image']
        boxes = transformed['bboxes']
        boxes = np.array([np.array([x1, y1, x2, y2]).round().astype(int) for x1, y1, x2, y2 in boxes])

        if verbose:
            dict_draw = draw_predictions_and_targets(img_originale, boxes)
            plt.imshow(dict_draw["img_pred"])
            plt.savefig("original")

    # Go back from x_min, y_min, x_max, y_max to x_min, y_min, w, h
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    # Format results as requested in the Evaluation tab
    return format_prediction_string(boxes, scores)


parser = argparse.ArgumentParser(
    description='Script to launch the training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--weight', type=str, help='weight file')
parser.add_argument('--detection_threshold',
                    type=float,
                    default=0.5,
                    help='detection_threshold')
parser.add_argument('--input_dir', type=str, help='input dir')
parser.add_argument('--resize', type=int, nargs='+', help='resize')
parser.add_argument('--verbose', type=bool, default=False, help='verbose')

args = parser.parse_args()

INPUT_DIR = args.input_dir
sys.path.insert(0, INPUT_DIR)
import greatbarrierreef

logger = init_logger("Trainer", "DEBUG")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device : {device}")

model_name = args.model
logger.info(f"Model : {model_name}")
model_cls = utils.import_class(model_name)
model = model_cls()
logger.info(f"Model : {model}")

pt_file = args.weight

model.to(device)
model.load_state_dict(torch.load(pt_file, map_location=device))
model.eval()

env = greatbarrierreef.make_env()
iter_test = env.iter_test()

detection_threshold = args.detection_threshold
logger.info(f"Detection threshold : {detection_threshold}")

for idx, (pixel_array, df_pred) in tqdm(
        enumerate(iter_test)):  # iterate through all test set images

    df_pred['annotations'] = predict(model, pixel_array, detection_threshold,
                                     device, args.resize, args.verbose)
    if args.verbose:
        print(df_pred['annotations'])

    env.predict(df_pred)