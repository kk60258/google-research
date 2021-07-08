from object_detection.architecture_search.create_config import get_configs_from_pipeline_file, write_config
from object_detection.architecture_search.search_strategy_config import iter_strategy
import os
import subprocess
import time
from absl import app
from absl import flags
import pathlib

FLAGS = flags.FLAGS

flags.DEFINE_string('base_config_path', '1', '')
flags.DEFINE_string('target_config_path_dir', '1', '')
flags.DEFINE_string('model_path_dir', '1', '')
flags.DEFINE_bool('keep_summary', False, '')
flags.DEFINE_integer('num_train_steps', 1, '')
# flags.DEFINE_integer('from_iter_num', 13, '')

#eval ap
flags.DEFINE_string('eval_workspace', '/home/jason/hic/workspace/', '')
flags.DEFINE_string('eval_ap_dir', 'eval_ap/', '')
flags.DEFINE_string('image_source_dir_hg', '/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_1920x1080_left_right_v1/*/', '')
flags.DEFINE_string('image_source_dir', '/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_300x300_left_right_v1/testing_0503/', '')

#parameters
flags.DEFINE_integer('code', 0, '')
flags.DEFINE_float('classification_weight', '1.0', '')
flags.DEFINE_float('sub_classification_weight', '1.0', '')
flags.DEFINE_float('localization_weight', '1.0', '')
flags.DEFINE_string('sub_classification_loss_class_weight', '1,2,1', '')
flags.DEFINE_float('learning_rate_base', 1e-2, '')
flags.DEFINE_float('warmup_learning_rate', 4e-4, '')
flags.DEFINE_integer('warmup_steps', 1000, '')
flags.DEFINE_integer('hold_base_rate_steps', 0, '')
flags.DEFINE_float('momentum_optimizer_value', 0.9, '')

flags.DEFINE_float('classification_loss_sigmoid_alpha', 0.75, '')
flags.DEFINE_float('classification_loss_sigmoid_gamma', 2, '')
flags.DEFINE_float('sub_classification_loss_sigmoid_alpha', 0.75, '')
flags.DEFINE_float('sub_classification_loss_sigmoid_gamma', 2, '')


def main(argv):
  del argv

  base_config_path = FLAGS.base_config_path


  config = get_configs_from_pipeline_file(base_config_path)
  timestamp_start = time.strftime("%m%d%H", time.localtime())
  target_config_path_dir = os.path.join(FLAGS.target_config_path_dir, timestamp_start)
  pathlib.Path(target_config_path_dir).mkdir(parents=True, exist_ok=True)
  model_path_dir = FLAGS.model_path_dir #os.path.join(FLAGS.model_path_dir, timestamp_start)
  pathlib.Path(model_path_dir).mkdir(parents=True, exist_ok=True)

  num_train_steps = FLAGS.num_train_steps

  strategy = {
              'classification_weight': FLAGS.classification_weight,
              'sub_classification_weight': FLAGS.sub_classification_weight,
              'localization_weight': FLAGS.localization_weight,
              'sub_classification_loss_class_weight': [float(s) for s in FLAGS.sub_classification_loss_class_weight.split(',')],
              'learning_rate_base': FLAGS.learning_rate_base,
              'warmup_learning_rate': FLAGS.warmup_learning_rate,
              'warmup_steps': FLAGS.warmup_steps,
              'hold_base_rate_steps': FLAGS.hold_base_rate_steps,
              'momentum_optimizer_value': FLAGS.momentum_optimizer_value,
              'classification_loss_sigmoid_alpha': FLAGS.classification_loss_sigmoid_alpha,
              'classification_loss_sigmoid_gamma': FLAGS.classification_loss_sigmoid_gamma,
              'sub_classification_loss_sigmoid_alpha': FLAGS.sub_classification_loss_sigmoid_alpha,
              'sub_classification_loss_sigmoid_gamma': FLAGS.sub_classification_loss_sigmoid_gamma,
              }

  for k, v in strategy.items():
    print(k, v)
  config.model.ssd.loss.classification_loss.weighted_sigmoid_focal.alpha = strategy['classification_loss_sigmoid_alpha']
  config.model.ssd.loss.classification_loss.weighted_sigmoid_focal.gamma = strategy['classification_loss_sigmoid_gamma']
  config.model.ssd.loss.sub_classification_loss.weighted_sigmoid_focal.alpha = strategy['sub_classification_loss_sigmoid_alpha']
  config.model.ssd.loss.sub_classification_loss.weighted_sigmoid_focal.gamma = strategy['sub_classification_loss_sigmoid_gamma']

  config.model.ssd.loss.classification_weight = strategy['classification_weight']
  config.model.ssd.loss.sub_classification_weight = strategy['sub_classification_weight']
  config.model.ssd.loss.localization_weight = strategy['localization_weight']
  del config.model.ssd.loss.sub_classification_loss_class_weight[:]
  sub_classification_loss_class_weight = [0]
  sub_classification_loss_class_weight.extend(strategy['sub_classification_loss_class_weight'])
  config.model.ssd.loss.sub_classification_loss_class_weight.extend(sub_classification_loss_class_weight)

  config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = strategy['learning_rate_base']
  config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = strategy['warmup_learning_rate']
  config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = strategy['warmup_steps']
  config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.hold_base_rate_steps = strategy['hold_base_rate_steps']
  config.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = strategy['momentum_optimizer_value']

  i = FLAGS.code

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
  eval_result_txt = os.path.join(model_path_dir, "eval_simple_result.txt")
  cmd = "python3 {} \
        -np \
        --ground_truth={} \
        --predict={} \
        --simple={}".format(os.path.join(FLAGS.eval_workspace, "eval_ap", "main_2classes.py"), ground_truth_path, eval_ap_dir, eval_result_txt)

  os.system(cmd)

  #check
  with open(eval_result_txt, 'r') as f:
    data = f.read()

  lines = data.split('\n')
  names = lines[0:-1:2]
  # for n in names:
  #   print('...' + n)

  if not names[-1] == eval_ap_dir:
    print("append error result due to last eval_simple_result {} vs current {}".format(names[-1], eval_ap_dir))
    simple_result = "tp 0, tp_sub 0, fp 0, total 0 precision 0, recall 0, " \
                    "accuracy_sub 0, accuracy_sub_per_tp 0, accuracy_arron_sub 0, " \
                    "f1 0, confidence_threshold 0," \
                    " raise_hand_recall 0, raise_hand_recall_arron 0"
    with open(eval_result_txt, 'a') as f:
      f.write(eval_ap_dir + '\n')
      f.write("{}, player_AP 0.00%\n".format(simple_result))

  cmd = "python3 {} {}".format(os.path.join(FLAGS.eval_workspace, "keep_last_ckpt.py"), model_dir)
  os.system(cmd)

  # subprocess.run("python3 object_detection/model_main.py \
  #           --model_dir=/tempssd/people_detection2/{} \
  #           --sample_1_of_n_eval_examples=10 \
  #           --pipeline_config_path={} \
  #           --num_train_steps=1".format(os.path.basename(target_config)[:-4], target_config)
  #           )


if __name__ == '__main__':
  app.run(main)
