from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_trials', type=int, default=3)
parser.add_argument('--base_config_path', type=str, default='/home/jason/github/google_obj_detection/research/object_detection/samples/jason/search_v1_from_061512.config')
parser.add_argument('--target_config_path_dir', type=str, default='/home/jason/github/google_obj_detection/research/object_detection/samples/jason')
parser.add_argument('--model_path_dir', type=str, default='/tempssd/people_detection2/0709_2')
parser.add_argument('--keep_summary', type=bool, default=False)
parser.add_argument('--num_train_steps', type=int, default=1)

parser.add_argument('--eval_workspace', type=str, default='/home/jason/hic/workspace/')
parser.add_argument('--eval_ap_dir', type=str, default='eval_ap/')
parser.add_argument('--image_source_dir_hg', type=str, default='/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_1920x1080_left_right_v1/*/')
parser.add_argument('--image_source_dir', type=str, default='/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_300x300_left_right_v1/testing_0503/')

parser.add_argument('--optimize', type=str, default='raise_hand_recall_arron')

args = parser.parse_args()

code = 0
def f(space):
  global code
  code += 1
  for sample in space.items():
    print(sample)

  model_path_dir = args.model_path_dir
  classification_weight = space['classification_weight']
  sub_classification_weight = space['sub_classification_weight']
  localization_weight = space['localization_weight']
  sub_classification_loss_class_weight = list(space['sub_classification_loss_class_weight'])

  learning_rate_base = space['learning_rate_base']
  warmup_learning_rate = space['warmup_learning_rate']
  warmup_steps = int(space['warmup_steps'])
  hold_base_rate_steps = int(space['hold_base_rate_steps'])
  momentum_optimizer_value = space['momentum_optimizer_value']
  classification_loss_sigmoid_alpha = space['classification_loss_sigmoid_alpha']
  classification_loss_sigmoid_gamma = space['classification_loss_sigmoid_gamma']
  sub_classification_loss_sigmoid_alpha = space['sub_classification_loss_sigmoid_alpha']
  sub_classification_loss_sigmoid_gamma = space['sub_classification_loss_sigmoid_gamma']
  if classification_weight < 0 or sub_classification_weight < 0 or localization_weight < 0 or any(sub_classification_loss_class_weight) < 0 or learning_rate_base < warmup_learning_rate:
    return {"loss": 12345, "status": STATUS_FAIL, "optimize": None}

  sub_classification_loss_class_weight = ','.join([str(s) for s in sub_classification_loss_class_weight])
  cmd = "python3 search_strategy_by_hyperopt.py \
        --code={} \
        --base_config_path={} \
        --target_config_path_dir={} \
        --model_path_dir={}\
        --keep_summary={} \
        --num_train_steps={} \
        --eval_workspace={} \
        --image_source_dir_hg={} \
        --image_source_dir={} \
        --classification_weight={} \
        --sub_classification_weight={} \
        --localization_weight={} \
        --sub_classification_loss_class_weight={} \
        --learning_rate_base={} \
        --warmup_learning_rate={} \
        --warmup_steps={} \
        --hold_base_rate_steps={} \
        --momentum_optimizer_value={} \
        --classification_loss_sigmoid_alpha={} \
        --classification_loss_sigmoid_gamma={} \
        --sub_classification_loss_sigmoid_alpha={} \
        --sub_classification_loss_sigmoid_gamma={}".format(code, args.base_config_path, args.target_config_path_dir,
                                                           args.model_path_dir, args.keep_summary, args.num_train_steps,
                                                           args.eval_workspace,
                                                           args.image_source_dir_hg, args.image_source_dir,
                                                           classification_weight,
                                                           sub_classification_weight, localization_weight,
                                                           sub_classification_loss_class_weight, learning_rate_base,
                                                           warmup_learning_rate, warmup_steps, hold_base_rate_steps,
                                                           momentum_optimizer_value, classification_loss_sigmoid_alpha,
                                                           classification_loss_sigmoid_gamma,
                                                           sub_classification_loss_sigmoid_alpha,
                                                           sub_classification_loss_sigmoid_gamma)
  print(cmd)
  os.system(cmd)

  try:
    eval_result_txt = os.path.join(model_path_dir, "eval_simple_result.txt")
    with open(eval_result_txt, 'r') as f:
      data = f.read()

    lines = data.split('\n')

    result = lines[1::2]
    info = [c.split(',') for c in result]
    last_sort_by = [float(c.split(' ')[-1]) for c in info[-1] if c.strip().startswith(args.optimize)]

    return {"loss": -last_sort_by[0], "status": STATUS_OK, "optimize": args.optimize}
  except:
    return {"loss": 123456, "status": STATUS_FAIL, "optimize": None}

space = {'classification_weight': hp.uniform('classification_weight', 0.0, 2),
         'sub_classification_weight': hp.normal('sub_classification_weight', 5, 3),
         'localization_weight': hp.uniform('localization_weight', 0, 10),
         'sub_classification_loss_class_weight': [hp.uniform('unraised', 0, 2), hp.uniform('raised', 0, 10), hp.uniform('staff', 0, 10)],
         'learning_rate_base': hp.loguniform('learning_rate_base', -9.21, -2.3),  # 1e-1 ~ 1e-4
         'warmup_learning_rate': hp.loguniform('warmup_learning_rate', -13.81, -4.6),  # 1e-6 ~ 1e-2
         'warmup_steps': hp.quniform('warmup_steps', 0, 5000, 1),  # step interval 1
         'hold_base_rate_steps': hp.choice('hold_base_rate_steps', [0] + [hp.quniform('hold_base_rate_steps_uniform', 0, 1000, 1)] * 3),  # 0.25 probability to choice 0
         'momentum_optimizer_value': hp.choice('momentum_optimizer_value', [0.9] + [hp.uniform('momentum_optimizer_value_uniform', 0.0, 1.0)] * 3),
         'classification_loss_sigmoid_alpha': hp.choice('classification_loss_sigmoid_alpha', [0.75] + [hp.uniform('classification_loss_sigmoid_alpha_uniform', 0.0, 1.0)] * 3),
         'classification_loss_sigmoid_gamma': hp.choice('classification_loss_sigmoid_gamma', [2] + [hp.uniform('classification_loss_sigmoid_alpha_gamma_uniform', 1, 5)] * 3),
         'sub_classification_loss_sigmoid_alpha': hp.choice('sub_classification_loss_sigmoid_alpha', [0.75] + [hp.uniform('sub_classification_loss_sigmoid_alpha_uniform', 0.0, 1.0)] * 3),
         'sub_classification_loss_sigmoid_gamma': hp.choice('sub_classification_loss_sigmoid_gamma', [2] + [hp.uniform('sub_classification_loss_sigmoid_alpha_gamma_uniform', 1, 5)] * 3),
         }

trials = Trials()

best = fmin(
  fn=f,  # "Loss" function to minimize
  space=space,  # Hyperparameter space
  algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
  trials=trials,
  max_evals=args.max_trials  # Perform 1000 trials
)

print("Found minimum after {} trials:".format(args.max_trials))
print(best)

def get_sorted_trial_result(trials):
  valid_trial_list = [trial for trial in trials]
  losses = [float(trial['result']['loss']) for trial in valid_trial_list]
  index_sorted_loss = [i for i, loss in sorted(enumerate(losses), key=lambda x: x[1])]
  trial_obj = [valid_trial_list[i] for i in index_sorted_loss]
  return index_sorted_loss, trial_obj

model_path_dir = args.model_path_dir
trials_txt = os.path.join(model_path_dir, "trials.txt")
import json
with open(trials_txt, 'w') as f:
  index_sorted_loss, sorted_trial_result = get_sorted_trial_result(trials)
  for idx, trial in zip(index_sorted_loss, sorted_trial_result):
    result = json.dumps(trial['result'], default=float)
    space = json.dumps(trial['misc']['vals'], default=float)
    f.write(str(idx+1) + '\n' + result + '\n' + space + '\n')

