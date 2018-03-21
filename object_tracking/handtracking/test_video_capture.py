from utils.detector_utils import WebcamVideoStream
from utils.handle_args import get_args
from utils import detector_utils
import tensorflow as tf
import numpy as np
import cv2
import datetime

def load_graphs():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('/home/testuser/tf_tut/obj_detection/handtracking/hand_inference_graph/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        hp_graph_def = tf.GraphDef()
        with tf.gfile.GFile('/home/testuser/tf_tut/tf_image_classifier/tf_files/retrained_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            hp_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(hp_graph_def, name='')
        return detection_graph
        # res = detection(detection_graph)
        # return res

def detection(detection_graph):
    with tf.Session(graph=detection_graph) as sess:
        if (frame is not None):
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # draw bounding boxes
            detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, frame,
                                             detection_graph, sess)

            # label, score = detector_utils.classify_object(frame, detection_graph, sess)

        return scores, boxes


if __name__ == '__main__':
    args = get_args()

    dg = load_graphs()

    video_capture = WebcamVideoStream(src=args.video_source,
                                  width=args.width,
                                  height=args.height).start()

    num_hands_detect = 2
    num_frames = 0
    start_time = datetime.datetime.now()
    while True:
        frame = video_capture.read()
        im_width, im_height = video_capture.size()

        # actual detection
        scores, boxes = detection(dg)

        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time

        if frame is not None:
            if args.display > 0:
                if args.fps > 0:
                    detector_utils.draw_fps_on_image(
                        "FPS : " + str(int(fps)), frame)
            cv2.imshow('Muilti - threaded Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video_capture.stop()
    cv2.destroyAllWindows()