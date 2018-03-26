import tensorflow as tf
import numpy as np
import cv2


class FrozenModel():
    """  Importing and running isolated TF graph """
    def __init__(self, score_thresh=0.3, num_hands=2):
        self.score_thresh = score_thresh
        self.num_hands_detect = num_hands

    def load_graph(self, model_path):
        self.model_path = model_path
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
            self.sess = tf.Session(graph=self.graph)

    def set_input(self, cv2image):
        self.image_np = cv2image
        self.im_height, self.im_width = image_np.shape[:2]

    def detect_objects(self):
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        self.boxes = np.squeeze(boxes)
        self.scores = np.squeeze(scores)

    def draw_box_on_image(self):
        for i in range(self.num_hands_detect):
            if (self.scores[i] > self.score_thresh):
                (left, right, top, bottom) = (self.boxes[i][1] * self.im_width,  self.boxes[i][3] * self.im_width,
                                              self.boxes[i][0] * self.im_height, self.boxes[i][2] * self.im_height)
                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))
                cv2.rectangle(self.image_np, p1, p2, (77, 255, 9), 3, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                cv2.putText(image_np,
                    '{:.2f}'.format(self.scores[i] ),
                    (int(left), int(top+5)),
                    font, font_scale, (0, 0, 255), font_thickness)


if __name__ == '__main__':
    g = FrozenModel(score_thresh=0.2, num_hands=10)
    g.load_graph('frozen_inference_graph.pb')

#    image_np = cv2.imread('test2.png')
    image_np = cv2.imread('/home/testuser/obj_det_git/image_classifier/t1.jpg')
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    g.set_input(image_np)

    # actual detection
    g.detect_objects()

    # draw bounding boxes
    g.draw_box_on_image()
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection_TF', image_np)
    cv2.imwrite('tf_result2.png', image_np)
    cv2.waitKey()
