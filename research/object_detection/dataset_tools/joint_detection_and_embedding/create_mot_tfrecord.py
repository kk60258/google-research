from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import contextlib2
import tensorflow.compat.v1 as tf
import pandas as pd
from object_detection.dataset_tools import oid_tfrecord_creation
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util

import six

from object_detection.core import standard_fields
from object_detection.utils import dataset_util
import cv2

tf.flags.DEFINE_string('image_dir', 'images', '')
tf.flags.DEFINE_string('label_dir', 'labels_with_ids', '')
tf.flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')
tf.flags.DEFINE_integer('image_size', 300, '')
tf.flags.DEFINE_string(
    'output_tf_record_path_prefix', '/tempssd/people_detection2/dataset/MOT17/',
    'Path to the output TFRecord. The shard index and the number of shards '
    'will be appended for each output shard.')
tf.flags.DEFINE_string('track_group', 'anonymous', '')
FLAGS = tf.flags.FLAGS

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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    required_flags = [
        'label_dir'
    ]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    image_paths, label_paths = get_all_files(FLAGS.label_dir)
    tf.logging.log(tf.logging.INFO, 'Total amount {} images and labels'.format(len(image_paths)))

    label_pbtxt_path = os.path.join('object_detection', 'data', 'oid_bbox_people_label_map.pbtxt')
    print(label_pbtxt_path)
    label_map = label_map_util.get_label_map_dict(label_pbtxt_path)

    track_group = FLAGS.track_group
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, FLAGS.output_tf_record_path_prefix,
            FLAGS.num_shards)
        image_counter = 0
        annotation_counter = 0
        max_instance_id = 0
        for counter, image_path_label_path in enumerate(zip(image_paths, label_paths)):
            image_path, label_path = image_path_label_path
            if not os.path.basename(image_path)[:-4] == os.path.basename(label_path)[:-4]:
                tf.logging.log(tf.logging.WARN, "not match image_path {} vs label_path {}".format(image_path, label_path))
                continue

            if not os.path.exists(image_path):
                tf.logging.log(tf.logging.WARN, "image_path {} not exist".format(image_path))
                continue

            with open(label_path) as f:
                raw = f.read()

            if len(raw) == 0:
                print('skip {}'.format(image_path))
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
            df['xmin'] = df['xcenter'] - df['width'] * 0.5
            df['xmin'].clip(lower=0.0, upper=1.0, inplace=True)

            df['xmax'] = df['xcenter'] + df['width'] * 0.5
            df['xmax'].clip(lower=0.0, upper=1.0, inplace=True)

            df['ymin'] = df['ycenter'] - df['height'] * 0.5
            df['ymin'].clip(lower=0.0, upper=1.0, inplace=True)

            df['ymax'] = df['ycenter'] + df['height'] * 0.5
            df['ymax'].clip(lower=0.0, upper=1.0, inplace=True)


            image_name = os.path.basename(image_path)
            tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', FLAGS.num_shards * 10, counter)
            num_of_annotations = len(df)
            tf.logging.log(tf.logging.INFO, "image {} with labels {}".format(image_name, num_of_annotations))
            # with tf.gfile.Open(image_path, 'rb') as image_file:
            #     encoded_image = image_file.read()
            #     image_decoded = tf.cond(
            #         tf.image.is_jpeg(encoded_image),
            #         lambda: tf.image.decode_jpeg(encoded_image, channels=3),
            #         lambda: tf.image.decode_png(encoded_image, channels=3))
            #     # image_decoded = tf.image.decode_image(encoded_image, channels=3)
            #     image_resize = tf.image.resize(image_decoded, (FLAGS.image_size, FLAGS.image_size))
            #     image_resize = tf.cast(image_resize, dtype=tf.uint8)
            #     encoded_image_resize = tf.io.encode_jpeg(image_resize, quality=100)
            image = cv2.imread(image_path)
            image_resize = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
            image_encode = cv2.imencode('.jpg', image_resize)[1]
            source_id = image_path
            tf_example = tf_example_from_annotations_data_frame(
                df, label_map, image_name, source_id, image_encode.tobytes(), track_group)

            instance_int = [int(x) for x in instance_id]
            if max(instance_int) > max_instance_id:
                max_instance_id = max(instance_int)

            if tf_example:
                if FLAGS.num_shards == 1:
                    shard_idx = 0
                else:
                    shard_idx = int(counter) % FLAGS.num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                image_counter += 1
                annotation_counter += num_of_annotations
        print('max instance id {}'.format(max_instance_id))
        print("tfrecord image counter {}, annotation counter {}".format(image_counter, annotation_counter))


# /m/01g317	Person
def tf_example_from_annotations_data_frame(df, label_map, image_name, source_id,
                                           encoded_image, track_group):
    feature_map = {
        standard_fields.TfExampleFields.object_bbox_ymin:
            dataset_util.float_list_feature(
                df['ymin'].to_numpy()),
        standard_fields.TfExampleFields.object_bbox_xmin:
            dataset_util.float_list_feature(
                df['xmin'].to_numpy()),
        standard_fields.TfExampleFields.object_bbox_ymax:
            dataset_util.float_list_feature(
                df['ymax'].to_numpy()),
        standard_fields.TfExampleFields.object_bbox_xmax:
            dataset_util.float_list_feature(
                df['xmax'].to_numpy()),
        standard_fields.TfExampleFields.object_track_label:
            dataset_util.int64_list_feature(
                df['instance_id'].to_numpy()),
        standard_fields.TfExampleFields.object_track_group:
            dataset_util.bytes_feature(six.ensure_binary(track_group)),
        standard_fields.TfExampleFields.object_class_text:
            dataset_util.bytes_list_feature([
                six.ensure_binary('/m/01g317')
                for _ in df['class_id'].to_numpy()
            ]),
        standard_fields.TfExampleFields.object_class_label:
            dataset_util.int64_list_feature([
                label_map['/m/01g317']
                for _ in df['class_id'].to_numpy()
            ]),
        standard_fields.TfExampleFields.filename:
            dataset_util.bytes_feature(
                six.ensure_binary(image_name)),
        standard_fields.TfExampleFields.source_id:
            dataset_util.bytes_feature(six.ensure_binary(source_id)),
        standard_fields.TfExampleFields.image_encoded:
            dataset_util.bytes_feature(six.ensure_binary(encoded_image)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_map))


if __name__ == '__main__':
    tf.app.run()