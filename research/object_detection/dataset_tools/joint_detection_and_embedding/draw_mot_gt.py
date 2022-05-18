import cv2
import numpy as np
import pandas as pd
import argparse
import os
import pathlib

argparse = argparse.ArgumentParser()
argparse.add_argument('--image_dir', type=str, default='/tempssd/people_detection2/dataset/MOT17/images/train/MOT17-02-SDP/img1')
argparse.add_argument('--gt_file', type=str, default='/tempssd/people_detection2/dataset/MOT17/images/train/MOT17-02-SDP/gt/gt.txt')
argparse.add_argument('--out_dir', type=str, default='/tempssd/people_detection2/dataset/MOT17/images/train/MOT17-02-SDP/img1_gt')
args = argparse.parse_args()

df = pd.read_csv(args.gt_file, names=['frame_id', 'instance_id', 'left', 'top', 'width', 'height', 'score', 'class', 'visibility'])
df = df.apply(pd.to_numeric)
print(len(df))
df = df[df['class'].isin([1, 2, 7])]
print(len(df))

def get_color(idx):
  idx = idx * 3
  color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

  return color

pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

for frame_id, frame_annotations in df.groupby('frame_id'):
  image_id = '{:06d}.jpg'.format(frame_id)
  image_path = os.path.join(args.image_dir, image_id)
  image = cv2.imread(image_path)

  for row, annotation in frame_annotations.iterrows():
    xmin = int(annotation['left'])
    ymin = int(annotation['top'])
    xmax = int(xmin + annotation['width'])
    ymax = int(ymin + annotation['height'])
    id = annotation['instance_id']
    color = get_color(id)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)
  image_out_path = os.path.join(args.out_dir, image_id)
  cv2.imwrite(image_out_path, image)


