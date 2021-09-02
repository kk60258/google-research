import numpy as np
import tensorflow as tf
import cv2
import time
import math
import pandas as pd
import pathlib
import os

from absl import flags
from absl import app
flags.DEFINE_string(
    'detect_saved_model', None, 'path of obj detection saved_model'
)

flags.DEFINE_string(
    'detect_tf', None, 'path of obj detection tflite'
)

flags.DEFINE_string(
    'image_source_dir', None, 'path of input video'
)

flags.DEFINE_string(
    'image_output_dir', None, 'dir of output videos'
)

flags.DEFINE_boolean(
    'output_oid', False, 'output detections in oid format or not'
)

FLAGS = flags.FLAGS

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (0, 125, 0), (0, 0, 125), (125, 0, 0), (0, 125, 125), (125, 0, 125), (125, 125, 0),
          (0, 200, 0), (0, 0, 200), (200, 0, 0), (0, 200, 200), (200, 0, 200), (200, 200, 0),
          (0, 175, 0), (0, 0, 175), (175, 0, 0), (0, 175, 175), (175, 0, 175), (175, 175, 0),
          (0, 100, 0), (0, 0, 100), (100, 0, 0), (0, 100, 100), (100, 0, 100), (100, 100, 0),
          (0, 50, 0), (0, 0, 50), (50, 0, 0), (0, 50, 50), (50, 0, 50), (50, 50, 0),]

SIZE = 300

# in sec
DETECTION_CONFIDENCE_THRESHOLD = 0.5
PEOPLE_CLASS = 1


class TfInterpreter(object):
    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, image_rgb):
        image_rgb = cv2.resize(image_rgb,(SIZE, SIZE))
        image_rgb = image_rgb[np.newaxis, ...]
        #np.array(image_rgb, dtype="uint8")
        image_rgb = np.array(image_rgb, dtype="float32") * (2.0 / 255.0) - 1.0

        interpreter = self.interpreter
        input_details = self.input_details
        output_details = self.output_details

        interpreter.set_tensor(input_details[0]['index'], image_rgb)
        interpreter.invoke()

        output_location = interpreter.get_tensor(output_details[0]['index'])
        output_class = interpreter.get_tensor(output_details[1]['index'])
        output_score = interpreter.get_tensor(output_details[2]['index'])
        output_num_detections = interpreter.get_tensor(output_details[3]['index'])

        return output_location, output_class, output_score, output_num_detections


class SavedModelInterpreter(object):
    def __init__(self, saved_model_dir, score_thres):
        self.output_name_list = ['detection_boxes',
                                 'detection_classes',
                                 'detection_scores',
                                 'num_detections'
                                 ]
        self.score_thres = score_thres
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        signature = meta_graph_def.signature_def
        serving_default = signature['serving_default']
        input_tensor_name = serving_default.inputs['inputs'].name
        output_tensor_name_list = [serving_default.outputs[name].name for name in self.output_name_list]

        self.sess = sess
        self.input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensors = [sess.graph.get_tensor_by_name(output_tensor_name) for output_tensor_name in output_tensor_name_list]

    def run(self, image):
        image = cv2.resize(image,(SIZE, SIZE))
        image = image[np.newaxis, ...] # normalize value in -1~+1 in model.preprocess
        outputs = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: image})
        output_location = outputs[0]
        output_class = outputs[1]
        output_score = outputs[2]
        output_num_detections = outputs[3]

        return output_location, output_class, output_score, output_num_detections


def detect_people(people_detection_interpreter, image_rgb, threshold=DETECTION_CONFIDENCE_THRESHOLD):

    output_location, output_class, output_score, output_num_detections = people_detection_interpreter.run(image_rgb)

    pred_bboxes = []
    pred_scores = []

    for i in range(int(output_num_detections[0])):
        if output_class[0, i] == PEOPLE_CLASS and output_score[0, i] > threshold:
            print("=======No {}=========".format(i))
            print("output_class {}".format(output_class[0, i]))
            print("output_score {}".format(output_score[0, i]))
            box = output_location[0, i]
            box = np.asarray([box[1], box[0], box[3], box[2]]) # transpose from[ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
            print("location {}".format(box))
            pred_bboxes.append(box)
            pred_scores.append(output_score[0, i])

    return pred_bboxes, pred_scores


def check_dir_exist(path):
    if not os.path.exists(path):
        check_dir_exist(os.path.dirname(path))
        os.mkdir(path)


def gointo_dir(path, result):
    if os.path.isdir(path):
        files = os.listdir(path)
        for f in files:
            file = os.path.join(path, f)
            if os.path.isdir(file):
                gointo_dir(file, result)
            else:
                a = file.split('.')[-1]
                if a != 'jpg':
                    continue
                result.append(file)
    elif path.split('.')[-1] == 'jpg':
        result.append(path)


def main(argv):
    if FLAGS.detect_saved_model is not None:
        people_detection_interpreter = SavedModelInterpreter(FLAGS.detect_saved_model, DETECTION_CONFIDENCE_THRESHOLD)
    elif FLAGS.detect_tf is not None:
        people_detection_interpreter = TfInterpreter(FLAGS.detect_tf)
    else:
        print('must provide saved model or tflite file')
        exit(1)

    image_source_dir = []
    if FLAGS.image_source_dir is not None:
        image_source_dir.append(FLAGS.image_source_dir)

    image_file_names = []
    for p in image_source_dir:
        gointo_dir(p, image_file_names)                        
    sorted(image_file_names)
    
    output_dir = FLAGS.image_output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_oid = FLAGS.output_oid
    if output_oid:
        output_oid_dir = os.path.join(FLAGS.image_source_dir, 'oid')
        pathlib.Path(output_oid_dir).mkdir(parents=True, exist_ok=True)
        to_oid = ["ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,Width,Height,InstanceID\n"]

    for i, file in enumerate(image_file_names):
        total_now = time.time()
        # ret, image_bgr = vidcap.read()
        image_bgr = cv2.imread(file)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_debug = image_bgr.copy()

        now = time.time()
        pred_bboxes, pred_scores = detect_people(people_detection_interpreter, image_rgb)
        then = time.time()

        print("Find person bbox in: {} sec".format(then - now))
        total_then = time.time()
        print("Total: {} sec".format(total_then - total_now))

        height = image_rgb.shape[0]
        width = image_rgb.shape[1]

        for id, result in enumerate(zip(pred_bboxes, pred_scores)):
            box, score = result
            if output_oid:
                image_id = os.path.basename(file)[:-4]
                data = "{},htc,/m/01g317,1,{},{},{},{},0,0,0,0,0," \
                       "{},{},-1\n".format(
                    image_id, box[0], box[1], box[2], box[3], width, height)
                to_oid.append(data)

            box = box * [width, height, width, height]
            box = [int(b) for b in box]
            if output_dir is not None:
                cv2.putText(image_debug, "{:.2f}".format(score), (box[0]+10, int((box[1] + box[3])/2)), cv2.FONT_HERSHEY_DUPLEX, 1, COLORS[id % len(COLORS)], 2, cv2.LINE_AA)
                cv2.rectangle(image_debug, (box[0], box[1]), (box[2], box[3]), color=COLORS[id % len(COLORS)], thickness=2)  # Draw Rectangle with the coordinates
        if output_dir is not None:
            output_file = os.path.join(output_dir, os.path.basename(file))
            cv2.imwrite(output_file, image_debug)

    with open(os.path.join(output_oid_dir, 'label.csv'), 'w') as f:
        for data in to_oid:
            f.write(data)

if __name__ == '__main__':
    app.run(main)