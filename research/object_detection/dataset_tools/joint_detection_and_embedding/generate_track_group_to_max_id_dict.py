import tensorflow.compat.v1 as tf
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import lookup as contrib_lookup

except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top

import json

'''
parse tfrecord
try dict loop up by tensor key
'''

input_files = ["/tempssd/people_detection2/dataset/MOT17/*tfrecord*"]
# input_files = ["/tempssd/people_detection2/dataset/MOT17/*tfrecord*",
#                "/tempssd/people_detection2/dataset/ETHZ/*tfrecord*",
#                "/tempssd/people_detection2/dataset/PRW/*tfrecord*",
#                "/tempssd/people_detection2/dataset/CUHK-SYSU/*tfrecord*",
#                "/tempssd/people_detection2/dataset/Citypersons/*tfrecord*",
#                "/tempssd/people_detection2/dataset/Caltech/*tfrecord*"]

with tf.Session() as sess:
  def get_int_list_value_from_feature(feature, key):
    return feature[key].int64_list.value

  def get_str_value_from_feature(feature, key):
    return feature[key].bytes_list.value[0].decode('utf-8')

  track_group_to_max_id_dict = {}
  total_annotations = 0
  total_images_set = set()
  total_images_list = list()
  total_images = 0

  for input_file in input_files:
    filenames = tf.gfile.Glob(input_file)
    tf.logging.info('Reading record datasets for input file: %s' % input_file)
    tf.logging.info('Number of filenames to read: %s' % len(filenames))
    for file in filenames:
      record_iterator = tf.python_io.tf_record_iterator(path=file)
      for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        image_name = get_str_value_from_feature(feature, 'image/filename')
        track_label_list = get_int_list_value_from_feature(feature, 'image/object/track/label')
        local_max_track_label = max(track_label_list)
        track_group = get_str_value_from_feature(feature, 'image/object/track/group')
        # print('-'*10)
        # print('{} {} {}'.format(track_group, local_max_track_label, image_name))

        # if image_name in total_images_set:
        #   print('duplicate {}, group {}'.format(image_name, track_group))
        total_images_set.add(image_name)
        total_images_list.append(image_name)
        total_images += 1
        total_annotations += len(track_label_list)
        last_id = track_group_to_max_id_dict.get(track_group, -2)
        if local_max_track_label > last_id:
          track_group_to_max_id_dict.update({track_group: local_max_track_label})

        # features = tf.parse_single_example(
        #   record,
        #   features={
        #     'image/filename':
        #       tf.FixedLenFeature((), tf.string, default_value=''),
        #     'image/object/track/group':
        #       tf.FixedLenFeature((), tf.string, default_value='anonymous'),
        #     'image/object/track/label':
        #       tf.FixedLenFeature([], tf.int64, default_value=0),
        #   }
        # )
        # track_group_t = features['image/object/track/group']
        # track_label_t = features['image/object/track/label']
        # # track_group = sess.run(track_group_t)
        # # track_label = sess.run(track_label_t)
        # print('{} {}'.format(track_group_t, track_label_t))
  print('images {}, annotations {}'.format(total_images, total_annotations))
  print('images list {}, set {}'.format(len(total_images_list), len(total_images_set)))
  print('-'*10)
  for k, v in track_group_to_max_id_dict.items():
    print(k, v)
  print('-'*10)
  with open('track_group_to_max_id_dict_only_mot.json', 'w') as f:
    f.write(json.dumps(track_group_to_max_id_dict))

  id_start = {}
  last = 0
  for k, v in track_group_to_max_id_dict.items():
    if v > 0:
      id_start.update({k: last})
      last = last + v
    else:
      id_start.update({k: 0})
  id_start.update({'total': last})

  for k, v in id_start.items():
    print(k, v)


'''
  try:
    # Dynamically try to load the tf v2 lookup, falling back to contrib
    lookup = tf.compat.v2.lookup
    hash_table_class = tf.compat.v2.lookup.StaticHashTable
  except AttributeError:
    lookup = contrib_lookup
    hash_table_class = contrib_lookup.HashTable

  sess.run(tf.initialize_all_tables())
  track_group_to_max_id_dict_t = hash_table_class(
    initializer=lookup.KeyValueTensorInitializer(
      keys=tf.constant(list(track_group_to_max_id_dict.keys())),
      values=tf.constant(list(track_group_to_max_id_dict.values()), dtype=tf.int64)),
    default_value=-1)

  input_tensor = tf.constant(['MOT17', 'MOT17'])

  lookup_result = track_group_to_max_id_dict_t.lookup(input_tensor)
  sess.run(tf.tables_initializer())

  print(sess.run(lookup_result))
'''