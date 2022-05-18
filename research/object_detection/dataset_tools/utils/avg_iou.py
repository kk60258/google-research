import numpy as np


def iou(nboxes, kboxes):
  '''
      nbox:numpy array,shape(n, 4,):xmin,ymin,xmax,ymax, or (4,)
          input box
      kboxes:numpy array,shape (k,4):xmin,ymin,xmax,ymax
          input ground truth boxes
          返回值：
       ious: numpy.array, shape (n, k)
  '''

  if nboxes.ndim == 2 and nboxes.shape[-1]:
    n = nboxes.shape[0]
    k = kboxes.shape[0]
    nboxes = nboxes[:, np.newaxis, :]
    # nboxes = np.tile(nboxes, (1, k, 1))
    kboxes = kboxes[np.newaxis, :, :]
    # kboxes = np.tile(kboxes, (n, 1, 1))

  box_area = (nboxes[..., 2] - nboxes[..., 0] + 1) * (nboxes[..., 3] - nboxes[..., 1] + 1)
  area = (kboxes[..., 2] - kboxes[..., 0] + 1) * (kboxes[..., 3] - kboxes[..., 1] + 1)
  xx1 = np.maximum(nboxes[..., 0], kboxes[..., 0])
  yy1 = np.maximum(nboxes[..., 1], kboxes[..., 1])
  xx2 = np.minimum(nboxes[..., 2], kboxes[..., 2])
  yy2 = np.minimum(nboxes[..., 3], kboxes[..., 3])

  w = np.maximum(0, xx2 - xx1 + 1)
  h = np.maximum(0, yy2 - yy1 + 1)

  inter = w * h
  ious = inter / (box_area + area - inter)
  return ious


def avg_iou(bboxes, clusters):
  max_ious =[np.max(iou(bboxes[i], clusters)) for i in range(bboxes.shape[0])]
  return np.mean(max_ious)