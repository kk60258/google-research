from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd
import argparse
import logging
from object_detection.dataset_tools.utils.kmeans_anchors import compute_centroids
from object_detection.dataset_tools.utils.dbscan_anchors import dbscan
argparse = argparse.ArgumentParser()

argparse.add_argument('--files', default='', type=str)
argparse.add_argument('--metric', default='iou', type=str)
argparse.add_argument('--cluster', default='kmeans', type=str)
args = argparse.parse_args()

def get_all_files(parent, img_collection=[], label_collection=[], extension='.txt'):
    for f in os.listdir(parent):
        child = os.path.join(parent, f)
        if os.path.isdir(child):
            get_all_files(child)
        elif f.endswith(extension):
            image_check_dir = parent.replace('labels_with_ids', 'images')
            if os.path.exists(os.path.join(image_check_dir, f.replace('.txt', '.jpg'))):
                img_collection.append(os.path.join(image_check_dir, f.replace('.txt', '.jpg')))
                label_collection.append(child)
            elif os.path.exists(os.path.join(image_check_dir, f.replace('.txt', '.png'))):
                img_collection.append(os.path.join(image_check_dir, f.replace('.txt', '.png')))
                label_collection.append(child)
    return img_collection, label_collection


def main():
    all_paths = args.files.split(',')

    if len(all_paths) == 0:
        logging.error('empty files')

    total_label_paths = []
    for path in all_paths:
        _, label_paths = get_all_files(path)
        total_label_paths.extend(label_paths)

    logging.log(logging.WARN, 'Total amount {} images and labels'.format(len(total_label_paths)))

    image_counter = 0
    annotation_counter = 0
    max_instance_id = 0
    total_annotations = []
    bboxes = []
    for counter, label_path in enumerate(total_label_paths):
        with open(label_path) as f:
            raw = f.read()

        if len(raw) == 0:
            print('skip {}'.format(label_path))
            continue

        raw = raw.replace('\n', ' ')
        raw = raw.split(' ')
        raw = [x for x in raw if x]
        class_id = raw[0:-1:6]
        instance_id = raw[1::6]
        bb_xcenter = raw[2::6]
        bb_ycenter = raw[3::6]
        bb_width = raw[4::6]
        bb_height = raw[5::6]
        df = pd.DataFrame(zip(class_id, instance_id, bb_xcenter, bb_ycenter, bb_width, bb_height),
                          columns=['class_id', 'instance_id', 'xcenter', 'ycenter', 'width', 'height'])
        df = df.apply(pd.to_numeric)
        # df['xcenter'].clip(lower=0.0, upper=1.0, inplace=True)
        # df['ycenter'].clip(lower=0.0, upper=1.0, inplace=True)
        # df['width'].clip(lower=0.0, upper=1.0, inplace=True)
        # df['height'].clip(lower=0.0, upper=1.0, inplace=True)
        df_bboxs = df[['xcenter', 'ycenter', 'width', 'height']]
        df_bboxs.clip(lower=0.0, upper=1.0, inplace=True)
        bboxes.extend(df_bboxs.to_numpy())

        num_of_annotations = len(df)

        image_counter += 1
        annotation_counter += num_of_annotations

    logging.log(logging.WARN, "image counter {}, annotation counter {}".format(image_counter, annotation_counter))

    if args.cluster == 'kmeans':
        compute_centroids(bboxes, k=6, iterations_num=100, metric=args.metric)
        # compute_centroids(bboxes, k=3, iterations_num=100, metric='aspect')
    elif args.cluster == 'dbscan':
        dbscan(bboxes, eps=0.3, min_samples=max(10, int(len(bboxes)*0.1)), metric=args.metric)





if __name__ == '__main__':
    main()