import tensorflow.compat.v1 as tf
import os
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def write_config(config, config_path):
  """Writes a config object to disk."""
  config_text = text_format.MessageToString(config)
  with tf.gfile.Open(config_path, "wb") as f:
    f.write(config_text)

def get_configs_from_pipeline_file(pipeline_config_path):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  return pipeline_config

if __name__ == '__main__':
  pipeline_config = get_configs_from_pipeline_file('/home/jason/github/google_obj_detection/research/object_detection/samples/jason/vivelanve_v2_gpu_noarron_sub_sigmoid.config')
  pipeline_config.model.ssd.num_classes = 10
  write_config(pipeline_config, '/home/jason/github/google_obj_detection/research/object_detection/samples/jason/0625.config')