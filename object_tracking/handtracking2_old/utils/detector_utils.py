# Utilities for object detector.
# try: https://stackoverflow.com/questions/42426960/how-does-one-train-multiple-models-in-a-single-script-in-tensorflow-when-there-a
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
import sys

from multiprocessing import Pool
import contextlib

image_classifier_path = r'/home/testuser/tf_tut/obj_detection'
sys.path.append(image_classifier_path)
from tf_graph_loader import ImportGraph
# from image_classifier.setup_nn_layers import make_nn_layers
# from image_classifier.settings import IMG_SIZE, CLASS_LABELS
# from tf_classifier.classify import init_pose_model, tf_classify


# detection_graph = tf.Graph()
# detection_graph_pose = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def my_model((path)):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess



# Load a frozen infrerence graph into memory

def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')
    #
    #     sess = tf.Session(graph=detection_graph)
    g = ImportGraph(PATH_TO_CKPT)
    print(">  ====== Hand Inference graph loaded.")
    return g

def load_inference_graph_old():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    # g = ImportGraph(PATH_TO_CKPT)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


def load_pose_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND POSE frozen graph into memory")
    # detection_graph_pose = tf.Graph()
    # with detection_graph_pose.as_default():
    #     od_graph_def_pose = tf.GraphDef()
    #     with tf.gfile.GFile("/home/testuser/PycharmProjects/obj_detector/test/tf_classifier/tf_files/retrained_graph.pb", 'rb') as fid:
    #         serialized_graph_pose = fid.read()
    #         od_graph_def_pose.ParseFromString(serialized_graph_pose)
    #         tf.import_graph_def(od_graph_def_pose, name='')
    #     sess_pose = tf.Session(graph=detection_graph_pose)
    f = ImportGraph('/home/testuser/tf_tut/tf_image_classifier/tf_files/retrained_graph.pb')
    print(">  ====== Hand Pose Inference graph loaded.")
    return f


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, f):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            scale_selection = 0.05
            left   -= scale_selection * left
            right  += scale_selection * right
            top    -= scale_selection * top
            bottom += scale_selection * bottom

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            # detect hand pose
            crop_img = process_subimg(image_np, (left, right, top, bottom))
            # guess, score = detect_hand_pose(crop_img, f.graph, f.sess)
            guess, score = (None, None)
            if guess is None:
                score = 0
                guess = '???'

            y = top - 5
            # label = '{:.1f}%:{}'.format(score*100, guess)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1

            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            # size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            # cv2.rectangle(image_np, p1, (int(left+size[0]), int(y-5-size[1])), (77, 255, 9), thickness=-1)
            # cv2.putText(image_np,
            #             label,
            #             (int(left), int(y)),
            #             font, font_scale, (0, 0, 0), font_thickness)


def process_subimg(image_np, box):
    (left, right, top, bottom) = box
    crop_img = image_np[int(top) : int(bottom),
                        int(left): int(right)]
    crop_img = cv2.resize(crop_img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img
    # gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # img_data = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    # data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    # model_out = classifier_model.predict([data])[0]

    # score = np.max(model_out)
    # guess = CLASS_LABELS[np.argmax(model_out)]
    # guess = ['Fist', 'Palm', 'C', 'Five', 'Point', 'V'][np.argmax(model_out)]


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 255, 9), 1)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, dg, s):
    # Definite input and output Tensors for detection_graph
    image_tensor = dg.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = dg.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = dg.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = dg.get_tensor_by_name(
        'detection_classes:0')
    num_detections = dg.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = s.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def detect_hand_pose(image_np, dg, s):

    softmax_tensor = dg.get_tensor_by_name('final_result:0')

    predictions = s.run(softmax_tensor, {'Mul:0': image_np})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # output
    for node_id in top_k:
        human_string = hand_pose_label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
        if score > 0.55:
            return human_string, score
        else:
            return None, None


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
