from hyperopt import pyll, hp

import pprint

pp = pprint.PrettyPrinter(indent=4, width=100)

# Define a complete space:
space = {
  'x': hp.normal('x', 0, 2),
  'y': hp.uniform('y', 0, 1),
  'use_float_param_or_not': hp.choice('use_float_param_or_not', [
    None, hp.uniform('float', 0, 1),
  ]),
  'my_abc_other_params_list': [
    hp.normal('a', 0, 2), hp.uniform('b', 0, 3), hp.choice('c', [False, True]),
  ],
  'yet_another_dict_recursive': {
    'v': hp.uniform('v', 0, 3),
    'u': hp.loguniform('w', -3, -1)
  }
}

# Print a few random (stochastic) samples from the space:
for _ in range(10):
  pp.pprint(pyll.stochastic.sample(space))
  # print(pyll.stochastic.sample(space))