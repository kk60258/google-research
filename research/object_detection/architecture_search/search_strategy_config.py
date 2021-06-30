__all__ = ['iter_strategy', 'max_index']

def frange(start, stop, interval):
  result = []
  while start < stop:
    result.append(start)
    start += interval
  return result

classification_weight_space = [0.1, 1]
sub_classification_weight_space = [1, 10]
localization_weight_space = [0.1, 1, 10]

sub_classification_loss_class_weight_no_raise_space = [0.1, 1]
sub_classification_loss_class_weight_raise_space = [1, 5, 10]
sub_classification_loss_class_weight_staff_space = [1, 5, 10]
# sub_classification_loss_class_weight_space = [sub_classification_loss_class_weight_no_raise_space, sub_classification_loss_class_weight_raise_space, sub_classification_loss_class_weight_staff_space]

# warmup_learning_rate = []
# warmup_steps = []
# hold_base_rate_steps = []

# search_config = [
#   ("classification_weight", classification_weight_space),
#   ("sub_classification_weight", sub_classification_weight_space),
#   ("localization_weight", localization_weight_space),
#   ("sub_classification_loss_class_weight", sub_classification_loss_class_weight_space)
# ]
#
# for k,v in search_config:
#   print("{} {} {}".format(k, len(v), v))

def max_index():
  return len(classification_weight_space) * len(sub_classification_weight_space) * len(localization_weight_space) * \
         len(sub_classification_loss_class_weight_no_raise_space) * len(sub_classification_loss_class_weight_raise_space) * len(sub_classification_loss_class_weight_staff_space)

def get_strategy(index):
  classification_weight = classification_weight_space[index % len(classification_weight_space)]
  index //= len(classification_weight_space)

  sub_classification_weight = sub_classification_weight_space[index % len(sub_classification_weight_space)]
  index //= len(sub_classification_weight_space)

  localization_weight = localization_weight_space[index % len(localization_weight_space)]
  index //= len(localization_weight_space)

  sub_classification_loss_class_weight_no_raise = sub_classification_loss_class_weight_no_raise_space[index % len(sub_classification_loss_class_weight_no_raise_space)]
  index //= len(sub_classification_loss_class_weight_no_raise_space)

  sub_classification_loss_class_weight_raise = sub_classification_loss_class_weight_raise_space[index % len(sub_classification_loss_class_weight_raise_space)]
  index //= len(sub_classification_loss_class_weight_raise_space)

  sub_classification_loss_class_weight_staff = sub_classification_loss_class_weight_staff_space[index % len(sub_classification_loss_class_weight_staff_space)]
  index //= len(sub_classification_loss_class_weight_staff_space)

  search_config = {
    "classification_weight": classification_weight,
    "sub_classification_weight": sub_classification_weight,
    "localization_weight": localization_weight,
    "sub_classification_loss_class_weight": [sub_classification_loss_class_weight_no_raise, sub_classification_loss_class_weight_raise, sub_classification_loss_class_weight_staff]
  }
  return search_config


def iter_strategy(start=0):
  print('max {}'.format(max_index()))
  for i in range(start, max_index()):
    r = get_strategy(i)
    yield i, r

if __name__ == '__main__':
  print(max_index())
  for i in iter_strategy():
    print(i)