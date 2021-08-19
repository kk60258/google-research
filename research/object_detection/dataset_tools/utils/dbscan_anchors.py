from sklearn.cluster import DBSCAN, OPTICS
import numpy as np
import matplotlib.pyplot as plt

# X = np.array([[1, 2], [2, 2], [2, 3],
#               [8, 7], [8, 8], [25, 80]])
# clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# print(clustering.labels_)
#
# print(clustering)

def distance_iou(box1, box2):
    # box1 and box2 are aligned, (x, y, width, height)
    iw = min(box1[2], box2[2]) * min(box1[3], box2[3])
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    iou = iw / (area1 + area2 - iw)
    return 1 - iou

def color_picker(num_group):
    if num_group < 9:
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        def pick_color(i):
            return color_map[i]
        return pick_color
    else:
        def pick_color(i):
            step = 1.0 / num_group
            return (i*step, 0.5, 0.5)

        return pick_color


def dbscan(bboxes, eps, min_samples, metric='iou'):
    bboxes = np.array(bboxes)
    if metric == 'iou':
        distance_fn = distance_iou
    else:
        distance_fn = 'euclidean'

    # clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_fn).fit(bboxes)
    clustering = OPTICS(min_samples=2).fit(bboxes)
    print(clustering.labels_)
    num_group = max(clustering.labels_) + 1
    groups = []
    for _ in range(num_group):
        groups.append([])

    print('num of groups is {}'.format(num_group))
    outliers = []
    for bbox_idx in range(clustering.labels_.shape[0]):
        group_idx = clustering.labels_[bbox_idx]
        if group_idx >= 0:
            groups[group_idx].append(bboxes[bbox_idx])
        else:
            outliers.append(bbox_idx)
    print('num of outliers is {}'.format(len(outliers)))

    centroids = []
    for boxes in groups:
        w = [box[2] for box in boxes]
        h = [box[3] for box in boxes]
        centroid_w = sum(w) / len(boxes)
        centroid_h = sum(h) / len(boxes)
        centroids.append([centroid_w, centroid_h])

    centroids_with_idx = sorted(enumerate(centroids), key=lambda x: x[1][0])
    for i, centroid in centroids_with_idx:
        print("dbscan result {}:".format(i))
        print(centroid[0], centroid[1])

    color = 0
    pick_color = color_picker(num_group)
    for boxes in groups:
        w = [box[2] for box in boxes]
        h = [box[3] for box in boxes]
        plt.scatter(w, h, s=10, color=pick_color(color))
        color +=1
    plt.show()


def show(w, h, cw, ch):
    import matplotlib.pyplot as plt
    plt.scatter(w, h, s=10, color='b')
    plt.scatter(cw, ch, s=10, color='r')
    plt.show()

if __name__ == '__main__':
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
    k = 7
    epsilon = 0.3
    min_samples = len(bboxes) * 0.1
    metric = 'iou'
    dbscan(bboxes, epsilon, min_samples, metric=metric)