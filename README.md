# Crown of Thorns starfish detection project by Clément Apavou & Guillaume Serieys

## Intro
Kaggle challenge TensorFlow - Help Protect the Great Barrier Reef. 
The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs.

![starfish](images/imageRD.png)

## Task suggested (to fill and use suggestion in the preproposal project)
### Handle with the dataset and metrics
     - [ ] Analyse dataset (split train/validation) (3 videos, find a split allowing to properly assess the ability of the model to generalize) 
     - [x] Dataloader Pytorch
     - [x] Script for metrics F2-score, Recall, Precision
Split possible: 
-    totally random (loss of temporality)
-    by videos: train: 0 & 1, val: 2 ?
-    by sequences
### To add (clément)
- [x] Class wandblogger:
     - [X] log all in wandblogger (checkpoint/weight, metrics, config)
     - [X] visual images predictions of training (correctly) afficher n dernières images à la fin de chaque batch
     - [ ] visual video sequence (validation (image dans l'ordre)
     - [x] torchmetrics pour les metrics?
- [X] loss regression, classif log dans wandb, IoU, F2score
- [X] early stopping
- [X] pour faster rcnn mettre "outputs" "preds" etc dictionnaire
- [x] scheduler
- [X] checkpoint plus propre (moniteur liée au metrics wandb)
- [X] enlever tout les trucs qui servent à rien (image classification)
- [X] option fast dv run si activé juste 2 batch train et val puis fin entrainement
- [x] data augmentation avec albumentation
- [ ] librairie detectron

Note: the test set keeps the temporality!\
"The API serves the images one by one, in order by video and frame number, as pixel arrays"
### Architectures 
     - [ ] Mask RCNN + CNN in backbone 
     - [ ] Yolo + CNN in backbone 
     - [ ] Mask RCNN + ViT in backbone 
     - [ ] Yolo + ViT in backbone
     - [ ] Yolo/Mask RCNN + Dynamic Head (head of classification with attention) + ViT in backbone
Maybe try (Sparse RCNN and others)
### Learning methods 
     - [ ] Unsupervised pre-training (Up-detr)
     - [ ] Self-supervised training for video (task: determine if the frames of a sequence are correctly ordonate)
     - [ ] Use external dataset for pre-training (Underwater dataset, Camouflaged dataset)

## Notes
- The competition metrics is the F2-score, so, tackle FN is more important than FP. So, False Positive are tolerate.
- All images in train_images have a size of 1280x720 (WxH)
- A video containes several sequences (split dataset by sequence ?) 

## Sequence repartition with bounding boxes
          Video_ID  Sequence  len_seq  nb BB
     0          0     40258      480    315
     1          0     45518      798    123
     2          0     59337      537    193
     3          0      8399     1423    896
     4          0     45015      617     24
     5          0     35305      853     89
     6          0     53708     1077   1146
     7          0       996      923    279
     8          1     60510     1167    113
     9          1     44160      151      0
     10         1     15827      770     74
     11         1     29424      184      0
     12         1      8503     2843   3195
     13         1     18048       71    115
     14         1     17665       87    255
     15         1     60754     2959   2632
     16         2     37114     2800      0
     17         2     26651     1525     29
     18         2     22643     1248   2349
     19         2     29859     2988     71

## Annotation details
<mark>video_id</mark> - ID number of the video the image was part of. The video ids are not meaningfully ordered.\
<mark>sequence</mark> - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.\
<mark>video_frame</mark> - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.\
<mark>sequence_frame</mark> - The frame number within a given sequence.\
<mark>image_id</mark> - ID code for the image, in the format '{video_id}-{video_frame}'\
<mark>annotations</mark> - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. Does not use the same format as the predictions you will submit. Not available in test.csv. A bounding box is described by the pixel coordinate (x_min, y_min) of its upper left corner within the image together with its width and height in pixels.

Competition BBOX = COCO format [x_min, y_min, width, height]

batch_size 10 with original size => 15.3Gb

Fix albumentation error in bbox_utils.py:
def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    bbox=list(bbox)
    for i in range(4):
        if (bbox[i]<0) :
            bbox[i]=0
        elif (bbox[i]>1) :
            bbox[i]=1
    bbox=tuple(bbox)
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
            raise ValueError(
                "Expected {name} for bbox {bbox} "
                "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))
