import numpy as np
import tensorflow as tf
import cv2
from utils import detector_utils

def detection(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        with tf.Session(graph=detection_graph) as sess:
            if (frame is not None):
                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)

    return boxes, scores


if __name__ == '__main__':
    # path = '/home/testuser/tf_tut/obj_detection/handtracking/hand_inference_graph/frozen_inference_graph.pb'
    imgpath = '/home/testuser/tf_tut/hand_pose_datasets/selected/palm/5_A_hgr2A1_id02_1.jpg'
    frame = cv2.imread(imgpath)

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

        with tf.Session(graph=detection_graph) as sess:
            if (frame is not None):
                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)

                image_np = cv2.resize(frame, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
                image_np = np.expand_dims(image_np, axis=0)
                softmax_tensor = detection_graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor, {'Mul:0': image_np})
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                for node_id in top_k:
                    human_str = ['palm', 'fist', 'point'][node_id]
                    score = predictions[0][node_id]
                    print human_str, score
