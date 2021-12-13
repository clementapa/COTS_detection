from os import listdir
import os, os.path as osp
from PIL import Image, ImageDraw
from tqdm import tqdm
import ast

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib import animation, rc
rc('animation', html='jshtml')

from utils.logger import init_logger
log_level = "DEBUG"
logger = init_logger("Data", log_level)

#### CSTE
CREATE_VIDEO = False

def validate_images(video_id):
    """
    Validate if there is corrupted data
    from https://www.kaggle.com/diegoalejogm/great-barrier-reefs-eda-with-animations
    """
    path = osp.join(path_videos, f"video_{video_id}")
    
    logger.info(f"Verifying that video {video_id} frames are valid...")
    for filename in tqdm(listdir(path), leave=True):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(osp.join(path,filename))
                img.verify() # Verify it is in fact an image
            except (IOError, SyntaxError) as e:
                logger.warning('Bad file:', filename) # Print out the names of corrupt files
    logger.info(f"Verified! Video {video_id} has all valid images")

def fetch_image_list(df_tmp, video_id, list_frame_id):
    """
    Load sequence of images with annotations
    """
    def fetch_image(frame_id):
        path_base = osp.join(path_videos, "video_{}/{}.jpg")
        raw_img = Image.open(path_base.format(video_id, frame_id))

        row_frame = df_tmp[(df_tmp.video_id == video_id) & (df_tmp.video_frame == frame_id)].iloc[0]
        bounding_boxes = ast.literal_eval(row_frame.annotations)

        for box in bounding_boxes:
            draw = ImageDraw.Draw(raw_img)
            x0, y0, x1, y1 = (box['x'], box['y'], box['x']+box['width'], box['y']+box['height'])
            draw.rectangle((x0, y0, x1, y1), outline=180, width=5)
        return raw_img

    return [np.array(fetch_image(index)) for index in list_frame_id]

def create_video(ims):
    """
    from https://www.kaggle.com/diegoalejogm/great-barrier-reefs-eda-with-animations
    """
    fig = plt.figure(figsize=(9, 9))
    plt.axis('off')
    im = plt.imshow(ims[0])

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//12)

# Init path 
root_path_data = "../tensorflow-great-barrier-reef/"
path_videos = osp.join(root_path_data, "train_images")
annotations = osp.join(root_path_data, "train.csv")

# Check annotations
df_train_raw = pd.read_csv(annotations)
logger.info(f"Train data :\n {df_train_raw}")

logger.info(f"Duplicate data : {df_train_raw.duplicated().sum()}")

nb_video = list(pd.unique(df_train_raw.video_id))

# Check images
for video_id in range(len(nb_video)):
    validate_images(video_id)

stats = []
# Retrieve images
for video_id in tqdm(range(len(nb_video)), position=0, desc="Video", leave=True):

    df_video = df_train_raw[df_train_raw.video_id == video_id]
    sequences = pd.unique(df_video.sequence)
    
    for seq in tqdm(sequences, position=1, desc="Sequence", leave=True):
        
        df_seq = df_video[df_video.sequence == seq]
        
        img_vid_frame = df_seq.video_frame 
        nb_BB = sum([len(ast.literal_eval(bb)) for bb in np.asarray(df_seq.annotations)])
        
        stats.append({"Video_ID": video_id, "Sequence": seq, "len_seq": len(img_vid_frame), "nb BB": nb_BB})
        # logger.info(f"Video ID: {video_id}, Sequence: {seq}, len sequence: {len(img_vid_frame)}, Nb BB: {nb_BB}")

        if CREATE_VIDEO:
            images = fetch_image_list(df_train_raw, video_id = video_id, list_frame_id = img_vid_frame)

            video = create_video(images)

            output_path = osp.join("video_annotations", f"video_{video_id}")
            os.makedirs(output_path, exist_ok=True)
            video.save(osp.join(output_path, f"sequence_{seq}.mp4"))
            logger.info(f"{osp.join(output_path, f'sequence_{seq}.mp4')} written !")

df_stats = pd.DataFrame(stats)
logger.info(f"Statistics database:\n {df_stats}")
