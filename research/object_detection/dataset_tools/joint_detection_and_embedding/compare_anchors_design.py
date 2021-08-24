from object_detection.dataset_tools.utils.h5py_helper import read_from_h5py
from object_detection.dataset_tools.joint_detection_and_embedding.analyze_mutliple_grid_anchors import analyze_ssd_anchors, analyze_grid_anchors
h5 = 'bboxes_mot.h5'
bboxes = read_from_h5py('bboxes', h5) # (N, 4), [xcenter, ycenter, width, height]
bboxes[:, :2] = 0 # align to (0, 0),
super_scales = []
super_aspects = []

min_scale = 0.2
max_scale = 0.9
num_layers = 6
scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
          for i in range(num_layers)]

scales += [1.0] # for scale_next
scales = sorted(scales)

aspects = [1.0, 1.0/2, 2.0, 1.0/3, 3.0]


super_scales.append(scales)
super_aspects.append(aspects)

super_scales.append([0.03, 0.06, 0.14, 0.39, 1])
super_aspects.append([0.19, 0.20, 0.21, 0.27])
'''
k-means iou result 0: w 0.01339210451784056, h 0.06544256900829908, ratio 0.20463904031857683, scale 0.029604285569409285
k-means iou result 1: w 0.026568147115217062, h 0.14074944796932637, ratio 0.18876199870430097, scale 0.061151059189801876
k-means iou result 2: w 0.04646750369642403, h 0.2246070870475557, ratio 0.20688351515188627, scale 0.10216129720997735
k-means iou result 3: w 0.09819467525619394, h 0.4604134670182052, ratio 0.21327498496544026, scale 0.2126267877700997
k-means iou result 4: w 0.26049713652944184, h 0.8778502120896294, ratio 0.29674440233870425, scale 0.47820232804861135
avg_iou 0.7343163858005904
'''
# super_scales.append([0.03, 0.06, 0.10, 0.21, 0.47, 1])
super_scales.append([0.03, 0.06, 0.10, 0.21, 1])
super_aspects.append([0.18, 0.20, 0.21, 0.29])
'''
--------------------------------------------------
k-means iou result 0: w 0.011847459619525942, h 0.056550778116775696, ratio 0.2095012661905604, scale 0.025884030988068133
k-means iou result 1: w 0.02050981031534969, h 0.10363854165083163, ratio 0.19789751948121043, scale 0.046104303818819616
k-means iou result 2: w 0.0290395006281257, h 0.154703443160825, ratio 0.18771075830508258, scale 0.06702619439325185
k-means iou result 3: w 0.048586236995909395, h 0.23406484020064944, ratio 0.20757597319725335, scale 0.10664112620559862
k-means iou result 4: w 0.100260254554866, h 0.46950449855072307, ratio 0.21354482196518154, scale 0.21696230211571366
k-means iou result 5: w 0.2633964001605989, h 0.8834480173982863, ratio 0.29814589537060615, scale 0.482386802795979
'''
# super_scales.append([0.02, 0.04, 0.06, 0.10, 0.21, 0.48, 1])
super_scales.append([0.02, 0.04, 0.06, 0.10, 0.21, 1])
super_aspects.append([0.18, 0.19, 0.20, 0.21, 0.29])

'''
--------------------------------------------------
k-means aspect result 0: 0.18519135781865778
k-means aspect result 1: 0.2591381343244203
avg_iou 0.07001167723162025
--------------------------------------------------
'''
super_aspects.append([0.18, 0.25])
'''
k-means aspect result 0: 0.17829882479728798
k-means aspect result 1: 0.2312189873965605
k-means aspect result 2: 0.33820356679098407
avg_iou 0.07392691473932077
--------------------------------------------------
'''
super_aspects.append([0.17, 0.23, 0.33])
'''
k-means aspect result 0: 0.16733516979273175
k-means aspect result 1: 0.20449045455372403
k-means aspect result 2: 0.25273465458637795
k-means aspect result 3: 0.36615566155409746
avg_iou 0.07753818035315391
--------------------------------------------------
'''
super_aspects.append([0.16, 0.20, 0.25, 0.36])
'''
k-means aspect result 0: 0.16193117560619685
k-means aspect result 1: 0.19360281917866368
k-means aspect result 2: 0.2267291880379222
k-means aspect result 3: 0.274049538527787
k-means aspect result 4: 0.3879547981312545
avg_iou 0.07964456912992157
--------------------------------------------------
'''
super_aspects.append([0.16, 0.19, 0.22, 0.27, 0.38])
'''
k-means aspect result 0: 0.15684615228258408
k-means aspect result 1: 0.18584035710139563
k-means aspect result 2: 0.2139502557086395
k-means aspect result 3: 0.25019934160895535
k-means aspect result 4: 0.30500087848967490.fv-
k-means aspect result 5: 0.41167983367008326
avg_iou 0.08162090337831387
'''
super_aspects.append([0.15, 0.18, 0.21, 0.25, 0.30, 0.41])

best_iou = 0
for scales in super_scales:
  for aspects in super_aspects:
    print('scales {}'.format(scales))
    print('aspects {}'.format(aspects))
    iou = analyze_ssd_anchors(bboxes, scales, aspects, reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=0)
    # iou = analyze_grid_anchors(bboxes, scales, aspects)
    print('-'*30)
    if iou > best_iou:
      best_iou = iou
      best = (scales, aspects)

best_scales, best_aspects = best
print('best scales {}'.format(best_scales))
print('best aspects {}'.format(best_aspects))
print('best iou %f'%best_iou)

