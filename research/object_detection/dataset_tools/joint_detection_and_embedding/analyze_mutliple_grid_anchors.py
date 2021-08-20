import numpy as np
import math
from object_detection.dataset_tools.utils.avg_iou import avg_iou

def analyze_ssd_anchors(scales, aspects, reduce_boxes_in_lowest_layer=True, interpolated_scale_aspect_ratio=1.0):
  anchors = []

  def scale_to_aspect(scale, aspect):
    ratio_sqrts = math.sqrt(aspect)
    height = scale / ratio_sqrts
    width = scale * ratio_sqrts
    return width, height

  for layer, (scale, scale_next) in enumerate(zip(scales[:-1], scales[1:])):
    if layer == 0 and reduce_boxes_in_lowest_layer:
      width, height = scale_to_aspect(0.1, 1.0)
      anchors.append([0, 0, width, height])
      for aspect in (2.0, 0.5):
        width, height = scale_to_aspect(scale, aspect)
        anchors.append([0, 0, width, height])
    else:
      for aspect in aspects:
        width, height = scale_to_aspect(scale, aspect)
        anchors.append([0, 0, width, height])

      if interpolated_scale_aspect_ratio > 1.0:
        interpolated_scale = math.sqrt(scale*scale_next)
        width, height = scale_to_aspect(interpolated_scale, interpolated_scale_aspect_ratio)
        anchors.append([0, 0, width, height])
  anchors = np.array(anchors)
  iou = avg_iou(bboxes, anchors)
  print('avg iou {}'.format(iou))

if __name__ == '__main__':
  # xcenter, ycenter, width, height
  bboxes = [
    [0.7403649999999999, 0.5625, 0.086979, 0.350926],
    [0.327344, 0.5356479999999999, 0.044271, 0.243519],
    [0.785417, 0.5546300000000001, 0.095833, 0.311111],
    [0.559375, 0.49907399999999996, 0.01875, 0.101852],
    [0.5763020000000001, 0.5013890000000001, 0.016146, 0.10648099999999999],
    [0.6622399999999999, 0.460185, 0.017188, 0.092593],
    [0.539583, 0.45185200000000003, 0.020833, 0.10740699999999999],
    [0.583333, 0.458333, 0.019791999999999997, 0.1],
    [0.497917, 0.456481, 0.021875, 0.105556],
    [0.257552, 0.543981, 0.054688, 0.26203699999999996],
    [0.34713499999999997, 0.510648, 0.031771, 0.173148],
    [0.723698, 0.45925900000000003, 0.026562, 0.11481500000000001],
    [0.786198, 0.45925900000000003, 0.032813, 0.11481500000000001],
    [0.269531, 0.541204, 0.046354, 0.23055599999999998],
    [0.294531, 0.47361099999999995, 0.018229, 0.08611100000000001],
    [0.228125, 0.46388900000000005, 0.020833, 0.077778],
    [0.31224, 0.483796, 0.018229, 0.12314800000000001],
    [0.514583, 0.45787, 0.016666999999999998, 0.071296],
    [0.30625, 0.419907, 0.010417000000000001, 0.039814999999999996],
    [0.315104, 0.416667, 0.009375, 0.038889],
  ]

  bboxes = [[max(bbox[1] - 0.5*bbox[3], 0), max(bbox[0] - 0.5*bbox[2], 0), min(bbox[1] + 0.5*bbox[3], 1), min(bbox[0] + 0.5*bbox[2], 1)] for bbox in bboxes]  # ymin, xmim, ymax, xmax
  bboxes = np.array(bboxes, dtype=np.float32)

  # scales = np.linspace(0.1, 0.9, num=6)
  min_scale = 0.2
  max_scale = 0.9
  num_layers = 6
  scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
            for i in range(num_layers)]

  scales += [1.0] # for scale_next
  scales = sorted(scales)

  aspects = [1.0, 1.0/2, 2.0, 1.0/3, 3.0]
  analyze_ssd_anchors(scales, aspects)



