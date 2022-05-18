import tensorflow.compat.v1 as tf
import numpy as np


class Model(object):
    def __init__(self, opt):
        self.output_name_list = ['raw_detection_boxes',
                                'raw_detection_scores',
                                'raw_track_embedding',
                                'detection_anchor_indices',
                                 ]
        self.score_thres = opt.score_thres
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], opt.saved_model_dir)
        signature = meta_graph_def.signature_def
        serving_default = signature['serving_default']
        input_tensor_name = serving_default.inputs['inputs'].name
        output_tensor_name_list = [serving_default.outputs[name].name for name in self.output_name_list]

        self.sess = sess
        self.input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensors = [sess.graph.get_tensor_by_name(output_tensor_name) for output_tensor_name in output_tensor_name_list]

    def run(self, image, original_height, original_width):
        # image = cv2.resize(image_rgb,(SIZE, SIZE))
        image = image[np.newaxis, ...] # normalize value in -1~+1 in model.preprocess
        outputs = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: image})
        raw_detection_boxes = outputs[0]
        raw_detection_scores = outputs[1]
        raw_track_embedding = outputs[2]
        detection_anchor_indices = outputs[3]

        pred_bboxes_list = []
        pred_scores_list = []
        pred_embeddings_list = []

        # pick nms result. Actually, we can get nms result directly.
        for no, idx in enumerate(detection_anchor_indices[0]):
            picked_detection_scores = raw_detection_scores[0, idx, 1]
            if picked_detection_scores < self.score_thres:
                break
            # [ymin, xmin, ymax, xmax]
            # value in 0-1
            picked_detection_boxes = raw_detection_boxes[0, idx]
            picked_detection_boxes = np.clip(picked_detection_boxes, 0, 1)
            # to absolute ordinate
            picked_detection_boxes = picked_detection_boxes * np.array([original_height, original_width, original_height, original_width])
            picked_detection_boxes = np.array([picked_detection_boxes[1], picked_detection_boxes[0], picked_detection_boxes[3], picked_detection_boxes[2]])
            picked_track_embedding = raw_track_embedding[0, idx]

            pred_bboxes_list.append(picked_detection_boxes)
            pred_scores_list.append(picked_detection_scores)
            pred_embeddings_list.append(picked_track_embedding)

        return pred_bboxes_list, pred_scores_list, pred_embeddings_list