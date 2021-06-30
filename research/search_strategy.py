from object_detection.architecture_search.create_config import get_configs_from_pipeline_file, write_config
from object_detection.architecture_search.search_strategy_config import iter_strategy
import os
import subprocess
import time
from absl import app
from absl import flags
import pathlib

FLAGS = flags.FLAGS

flags.DEFINE_string('base_config_path', '/home/jason/github/google_obj_detection/research/object_detection/samples/jason/search_v1_vivelanve_v2_gpu_noarron_sub_sigmoid.config', '')
flags.DEFINE_string('target_config_path_dir', '/home/jason/github/google_obj_detection/research/object_detection/samples/jason', '')
flags.DEFINE_string('model_path_dir', '/tempssd/people_detection2/', '')
flags.DEFINE_bool('keep_summary', False, '')
flags.DEFINE_integer('num_train_steps', 1, '')

#eval ap
flags.DEFINE_string('eval_workspace', '/home/jason/hic/workspace/', '')
flags.DEFINE_string('eval_ap_dir', 'eval_ap/', '')
flags.DEFINE_string('image_source_dir_hg', '/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_1920x1080_left_right_v1/*/', '')
flags.DEFINE_string('image_source_dir', '/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_300x300_left_right_v1/testing_0503/', '')

def main(argv):
  del argv

  base_config_path = FLAGS.base_config_path


  config = get_configs_from_pipeline_file(base_config_path)
  timestamp_start = time.strftime("%m%d%H", time.localtime())
  target_config_path_dir = os.path.join(FLAGS.target_config_path_dir, timestamp_start)
  pathlib.Path(target_config_path_dir).mkdir(parents=True, exist_ok=True)
  model_path_dir = os.path.join(FLAGS.model_path_dir, timestamp_start)
  pathlib.Path(model_path_dir).mkdir(parents=True, exist_ok=True)

  num_train_steps = FLAGS.num_train_steps
  for i, strategy in enumerate(iter_strategy()):
    config.model.ssd.loss.classification_weight = strategy['classification_weight']
    config.model.ssd.loss.sub_classification_weight = strategy['sub_classification_weight']
    config.model.ssd.loss.localization_weight = strategy['localization_weight']
    del config.model.ssd.loss.sub_classification_loss_class_weight[:]
    sub_classification_loss_class_weight = [0]
    sub_classification_loss_class_weight.extend(strategy['sub_classification_loss_class_weight'])
    config.model.ssd.loss.sub_classification_loss_class_weight.extend(sub_classification_loss_class_weight)

    config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 5e-2
    config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = 1e-4
    config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = 1000
    config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.hold_base_rate_steps = 300


    timestamp = time.strftime("%m%d%H", time.localtime())
    train_config_path_name ="{}_{}.config".format(timestamp, i)
    train_config_path = os.path.join(target_config_path_dir, train_config_path_name)
    write_config(config, train_config_path)
    model_dir = "{}/{}".format(model_path_dir, train_config_path_name[:-7])
    os.system("python3 object_detection/model_main.py "
              "--model_dir={} "
              "--pipeline_config_path={} "
              "--save_summary_steps=0 "
              "--keep_checkpoint_max=1 "
              "--num_train_steps={}".format(model_dir, train_config_path, num_train_steps)
              )

    os.system("rm {}/events.out*".format(model_dir))
    os.system("rm {}/graph.pbtxt".format(model_dir))

    cmd = "python3 object_detection/export_inference_graph.py \
          --input_type=image_tensor \
          --pipeline_config_path={} \
          --trained_checkpoint_prefix={} \
          --output_directory={}".format(os.path.join(model_dir, train_config_path_name), os.path.join(model_dir, "model.ckpt-{}".format(num_train_steps)), os.path.join(model_dir, "snpe"))
    os.system(cmd)

    eval_name = "predicted-viveland-{}".format(train_config_path_name[:-7])
    eval_ap_dir= os.path.join(FLAGS.eval_workspace, FLAGS.eval_ap_dir, eval_name)
    cmd = "python3 {} \
          --savedmodel_dir={} \
          --output_dir={} \
          --image_source_dir_hg={} \
          --image_source_dir={} \
          --output_txt=True \
          --output_img=False".format(os.path.join(FLAGS.eval_workspace, "eval_ap", "run_savedmodel_eval.py"),
                                     os.path.join(model_dir, "snpe", "saved_model"),
                                     eval_ap_dir, FLAGS.image_source_dir_hg, FLAGS.image_source_dir)
    os.system(cmd)

    ground_truth_path = os.path.join(FLAGS.eval_workspace, FLAGS.eval_ap_dir, 'ground-truth-viveland-0503_class2_v2')
    cmd = "python3 {} \
          -np \
          --ground_truth={} \
          --predict={} \
          --simple={}".format(os.path.join(FLAGS.eval_workspace, "eval_ap", "main_2classes.py"), ground_truth_path, eval_ap_dir, os.path.join(model_path_dir, "eval_simple_result.txt"))

    os.system(cmd)

    # subprocess.run("python3 object_detection/model_main.py \
    #           --model_dir=/tempssd/people_detection2/{} \
    #           --sample_1_of_n_eval_examples=10 \
    #           --pipeline_config_path={} \
    #           --num_train_steps=1".format(os.path.basename(target_config)[:-4], target_config)
    #           )


if __name__ == '__main__':
  app.run(main)
