from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

space = {'classification_weight': hp.uniform('classification_weight', 0, 2),
         'sub_classification_weight': hp.uniform('sub_classification_weight', 0, 6),
         'localization_weight': hp.uniform('localization_weight', 5, 10),
         'sub_classification_loss_class_weight': [hp.uniform('unraised', 1.5, 3), hp.uniform('raised', 2, 6), hp.uniform('staff', 2, 6)],
         'learning_rate_base': hp.loguniform('learning_rate_base', -5, -2.3),  # 1e-1 ~ 6.7e-3
         'warmup_learning_rate': hp.loguniform('warmup_learning_rate', -13.81, -10),  # 1e-6 ~ 4.5e-5
         'warmup_steps': hp.quniform('warmup_steps', 1000, 3000, 1),  # step interval 1
         'hold_base_rate_steps': hp.choice('hold_base_rate_steps', [0] + [hp.quniform('hold_base_rate_steps_uniform', 800, 1000, 1)] * 3),  # 0.25 probability to choice 0
         'momentum_optimizer_value': 0.9,
         'classification_loss_sigmoid_alpha': 0.75,
         'classification_loss_sigmoid_gamma': 2,
         'sub_classification_loss_sigmoid_alpha': 0.75,
         'sub_classification_loss_sigmoid_gamma': 2,
         }