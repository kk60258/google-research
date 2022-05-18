# coding=utf-8
# https://github.com/PaulChongPeng/darknet/blob/master/tools/k_means_yolo.py
# k-means ++ for YOLOv2 anchors
# 通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
import numpy as np
import math

# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def box_avg_iou(bboxes, centroids):
    max_iou_sum = 0
    for bbox in bboxes:
        max_iou = 0
        for centroid in centroids:
            iou = box_iou(bbox, centroid)
            max_iou = max(max_iou, iou)
        max_iou_sum += max_iou
    return max_iou_sum / len(bboxes)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def distance_builder(metric):
    if metric == 'iou':
        def iou_distance(box, centroid):
            return 1 - box_iou(box, centroid)
        return iou_distance
    elif metric == 'aspect':
        def aspect_distance(box, centroid):
            theta1 = math.atan(box.w / box.h)
            theta2 = math.atan(centroid.w / centroid.h)
            return (theta1 - theta2) ** 2
        return aspect_distance

def init_centroids(boxes, n_anchors, metric):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num)
    centroids.append(boxes[centroid_index])

    print(centroids[0].w, centroids[0].h)
    distance_fn = distance_builder(metric)
    for centroid_index in range(0, n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = distance_fn(box, centroid)
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids, metric):
    loss = 0
    groups = []
    distance_fn = distance_builder(metric)
    update_fn = update_builder(metric)
    for i in range(n_anchors):
        groups.append([])

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = distance_fn(box, centroid)
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance

    new_centroids = update_fn(n_anchors, groups)

    return new_centroids, groups, loss


def update_builder(metric):
    if metric == 'iou':
        return update_centroid_by_wh
    elif metric == 'aspect':
        return update_centroid_by_aspect


def update_centroid_by_wh(n_anchors, groups):
    new_centroids = []

    for group_index in range(n_anchors):
        group = groups[group_index]
        new_centroid = Box(0, 0, 0, 0)
        for box in group:
            new_centroid.w += box.w
            new_centroid.h += box.h
        new_centroid.w /= len(groups[group_index])
        new_centroid.h /= len(groups[group_index])
        new_centroids.append(new_centroid)
    return new_centroids


def update_centroid_by_aspect(n_anchors, groups):
    new_centroids = []

    for group_index in range(n_anchors):
        group = groups[group_index]
        new_centroid = Box(0, 0, 0, 0)
        theta = sum([math.atan(box.w / box.h) for box in group])
        theta_avg = theta / len(group)

        new_centroid.w = math.tan(theta_avg)
        new_centroid.h = 1
        new_centroids.append(new_centroid)
    return new_centroids

    # 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(bboxes, k, loss_convergence=1e-6, iterations_num=100, plus=True, metric='iou'):
    """
    :param k:
    :param bboxes: a list of [xcenter, ycenter, width, height] ndarray or list
    :param loss_convergence:
    :param iterations_num:
    :param plus:
    :return:
    """
    boxes = []

    for bbox in bboxes:
        boxes.append(Box(0, 0, bbox[2], bbox[3]))

    del bboxes

    if plus:
        centroids = init_centroids(boxes, k, metric)
    else:
        centroid_indices = np.random.choice(len(boxes), k)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(k, boxes, centroids, metric)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(k, boxes, centroids, metric)
        iterations = iterations + 1
        print("%d loss = %f" % (iterations, loss))
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

    if metric == 'iou':
        centroids = sorted(centroids, key=lambda c: c.w)
        avg_iou = box_avg_iou(boxes, centroids)
        # print result
        for i, centroid in enumerate(centroids):
            print("k-means result {}:".format(i))
            print(centroid.w, centroid.h)
        print("avg_iou {}:".format(avg_iou))

        with open('kmeans_result.txt', 'a') as f:
            for i, centroid in enumerate(centroids):
                f.write("k-means iou result {}: w {}, h {}, ratio {}, scale {}\n".format(i, centroid.w, centroid.h, centroid.w/centroid.h, math.sqrt(centroid.w * centroid.h)))
            f.write("avg_iou {}\n".format(avg_iou))
            f.write("{}\n".format('-'*50))

        w = [box.w for box in boxes]
        h = [box.h for box in boxes]
        cw = [box.w for box in centroids]
        ch = [box.h for box in centroids]
        show(w, h, cw, ch)
    elif metric == 'aspect':
        # print result
        centroids = sorted(centroids, key=lambda c: c.w)
        avg_iou = box_avg_iou(boxes, centroids)
        for i, centroid in enumerate(centroids):
            print("k-means aspect result {}:".format(i))
            print(centroid.w / centroid.h)
        print("avg_iou {}:".format(avg_iou))
        with open('kmeans_result.txt', 'a') as f:
            for i, centroid in enumerate(centroids):
                f.write("k-means aspect result {}: {}\n".format(i, centroid.w / centroid.h))
            f.write("avg_iou {}\n".format(avg_iou))
            f.write("{}\n".format('-'*50))

        w = [math.atan(box.w/box.h) for box in boxes]
        h = [1 for _ in boxes]
        cw = [math.atan(box.w/box.h) for box in centroids]
        ch = [1 for _ in centroids]
        show(w, h, cw, ch)


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
    loss_convergence = 1e-6
    grid_size = 13
    iterations_num = 100
    plus = 1
    metric = 'iou'
    compute_centroids(bboxes, k, loss_convergence, iterations_num, plus, metric=metric)