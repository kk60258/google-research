from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

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