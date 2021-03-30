import tensorflow as tf
import os
def count_tfrecord_examples(
        tfrecords_dir: str,
) -> int:
    """
    Counts the total number of examples in a collection of TFRecord files.

    :param tfrecords_dir: directory that is assumed to contain only TFRecord files
    :return: the total number of examples in the collection of TFRecord files
        found in the specified directory
    """

    count = 0
    for file_name in os.listdir(tfrecords_dir):
        tfrecord_path = os.path.join(tfrecords_dir, file_name)
        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))

    return count

print(count_tfrecord_examples('/tempssd/people_detection2/dataset/coco2017_train_person_no_crowd'))
