import numpy as np
def bbox_iou_efficient(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_iou_quick(nboxes, kboxes):
    '''
        nbox:numpy array,shape(n, 4,):x1,y1,x2,y2,
            input box
        kboxes:numpy array,shape (k,4):x1,y1,x2,y2
            input ground truth boxes
            返回值：
         ious: numpy.array, shape (n, k)
    '''
    # n = nboxes.shape[0]
    # k = kboxes.shape[0]
    nboxes = nboxes[:, np.newaxis, :]
    # nboxes = np.tile(nboxes, (1, k, 1))
    kboxes = kboxes[np.newaxis, :, :]
    # kboxes = np.tile(kboxes, (n, 1, 1))

    box_area = (nboxes[:, :, 2] - nboxes[:, :, 0] + 1) * (nboxes[:, :, 3] - nboxes[:, :, 1] + 1)
    area = (kboxes[:, :, 2] - kboxes[:, :, 0] + 1) * (kboxes[:, :, 3] - kboxes[:, :, 1] + 1)
    xx1 = np.maximum(nboxes[:, :, 0], kboxes[:, :, 0])
    yy1 = np.maximum(nboxes[:, :, 1], kboxes[:, :, 1])
    xx2 = np.minimum(nboxes[:, :, 2], kboxes[:, :, 2])
    yy2 = np.minimum(nboxes[:, :, 3], kboxes[:, :, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ious = inter / (box_area + area - inter)
    return ious

if __name__ == '__main__':
    nboxes_list = [np.array([0, 0.1, 0.5, 0.8]), np.array([0.2, 0.5, 0.8, 0.9])] * 100
    kboxes_list = [np.array([0.2, 0.3, 0.6, 0.4]), np.array([0.3, 0.2, 0.4, 0.9]), np.array([0.4, 0.0, 0.6, 0.3])] * 90
    nboxes = np.stack(nboxes_list, axis=0) * 100
    kboxes = np.stack(kboxes_list, axis=0) * 100
    import time
    start = time.time()
    ious = bbox_iou_quick(nboxes, kboxes)
    end = time.time() - start
    print(ious)
    print(end)
    print('-' * 10)
    start = time.time()
    ious2 = bbox_iou_efficient(nboxes, kboxes)
    end = time.time() - start
    print(ious2)
    print(end)
