'''

copy from fall detection code base: tfrecord_coco_gym_shard.py
'''
import tensorflow as tf
import os, io
from object_detection.utils import dataset_util
from PIL import Image
import cv2
import numpy as np
flags = tf.app.flags
# flags.DEFINE_string('output_path', '/tempssd/people_detection/0908_people_detection_test_frame_diff_additional.record', 'Path to output TFRecord')
flags.DEFINE_string('output_dir', '/tempssd/people_detection/', '')
flags.DEFINE_string('output_name', '0803_coco_people_detection_train', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# path = '/home/paul_huang/HTC_AI/code_base/coco/train_person_images_1'
path = '/tempssd/people_detection/0803_train/img600x600'
# path = '/home/jason/Downloads/HumanFallDetectionDataset/multi_camera_pick/'
# path = '/tempssd/people_detection/people_testing_0721/img'
def create_tf_example(filename, image, width, height, xmins, ymins, xmaxs, ymaxs, total_instance):
    # TODO(user): Populate the following variables from your example.
    total_instance = int(total_instance)
    image_format = b'jpg'
    if total_instance == 1:
        classes_text = [b'people'] # List of string class name of bounding box (1 per box)
        classes = [1] # List of integer class id of bounding box (1 per box)
    else:
        classes_text = [b'people']
        classes = [1]
        for i in range(0, total_instance - 1):
            classes_text.append(b'people')
            classes.append(1)
    filename = tf.compat.as_bytes(filename)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    shard_count = 0
    total_shard = 100
    amount_per_shard = 5000
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, FLAGS.output_name + '_%.5d-of-%.5d' % (shard_count, total_shard) + '.record'))
    # TODO(user): Write code to read in your dataset to examples variable
    # with open('/tempssd/people_detection/people_training_coco_multi_0803.csv', 'r') as fp:
    with open('/tempssd/people_detection/people_training_coco_multi_0803_without_any_test.csv', 'r') as fp:
    # with open('/tempssd/people_detection/people_testing_0721.csv', 'r') as fp:
    # with open('/home/jason/Downloads/HumanFallDetectionDataset/merged_0701_total.csv', 'r') as fp:
    # with open('/home/jason/hic/HumanFallModel/fall_training/merged_0701_total.csv', 'r') as fp:
        all_lines = fp.readlines()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    total_images = 0
    delete_instance = 0
    last_filename = None
    image = None
    # name_set = set()
    for id, lines in enumerate(all_lines):
        filename = lines.strip().split(",")[0].split("/")[-1]
        if '000000' in filename:
            print(filename)
            continue

        if 'multi_camera_pick' in path:
            if 'chute' not in filename:
                print(filename)
                continue

            parsed_filename = filename.split('_')

            level1_dir = 'pick' + parsed_filename[0][-2:]
            level2_dir = parsed_filename[1]
            new_path = os.path.join(path, level1_dir, level2_dir)
        else:
            new_path = path
        # name_set.add(filename)
        # if len(name_set) % 1000 > 5:
        #     continue

        full_filename = os.path.join(new_path, '{}'.format(filename))

        if not os.path.exists(full_filename):
            print("not find {}".format(full_filename))
            continue

        if last_filename != filename:
            # write previous record
            if image is not None:
                tf_example = create_tf_example(last_filename, encode_image, image.shape[1], image.shape[0], xmins, ymins, xmaxs, ymaxs, len(xmins))
                writer.write(tf_example.SerializeToString())
                total_images = total_images + 1

                if total_images % amount_per_shard == 0:
                    writer.close()
                    shard_count += 1
                    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, FLAGS.output_name + '_%.5d-of-%.5d' % (shard_count, total_shard) + '.record'))

            # if image is not None:
            #     cv2.namedWindow(last_filename, cv2.WINDOW_NORMAL)
            #     cv2.moveWindow(last_filename, 600, 100)
            #     cv2.resizeWindow(last_filename, 600, 600)
            #     for index in range(len(xmins)):
            #         x1 = int(xmins[index] * image.shape[1])
            #         x2 = int(xmaxs[index] * image.shape[1])
            #         y1 = int(ymins[index] * image.shape[0])
            #         y2 = int(ymaxs[index] * image.shape[0])
            #         cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            #     cv2.imshow(last_filename, image)
            #     key = cv2.waitKey(0)
            #     if key == 27:
            #         exit()
            #     cv2.destroyWindow(last_filename)

            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            delete_instance = 0
            image = cv2.imread(full_filename)
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST)
            encode_image = cv2.imencode('.jpg', image)[1].tostring()
            last_filename = filename


        xmin = float(lines.strip().split(",")[6])
        ymin = float(lines.strip().split(",")[7])
        xmax = float(lines.strip().split(",")[8])
        ymax = float(lines.strip().split(",")[9])
        init_total_instance = int(lines.strip().split(",")[5])
        which_instance = int(lines.strip().split(",")[4])
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        threshold = 1

        #cv2.rectangle(frame_sub, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #print("{}, num {}, valid pixel {}, total pixel {}".format(filename, len(xmins), bbox_valid_count, (y2 - y1) * (x2 - x1)))
        if area <= threshold:
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
        else:
            delete_instance = delete_instance + 1


    #final image
    # tf_example = create_tf_example(last_filename, encode_image, additional_encode_image, image.shape[1], image.shape[0], xmins, ymins, xmaxs, ymaxs, len(xmins))
    # writer.write(tf_example.SerializeToString())
    # total_images = total_images + 1
    print("total_images", total_images)

    writer.close()
    for i in range(shard_count + 1):
        os.rename(os.path.join(FLAGS.output_dir, FLAGS.output_name + '_%.5d-of-%.5d' % (i, total_shard) + '.record'),
                  os.path.join(FLAGS.output_dir, FLAGS.output_name + '_%.5d-of-%.5d' % (i, shard_count) + '.record'))

if __name__ == '__main__':
    tf.app.run()