import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from threading import Thread


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow warnings


class FrozenGraph(object):
    def __init__(self, score_thresh=0.2, num_hands_detect=2):
        self.score_thresh = score_thresh
        self.num_hands_detect = num_hands_detect
        self.label_lines = ['palm', 'fist', 'point']

    def load_graph(self, pbpath):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pbpath, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

    def set_input(self, cv2image):
        self.image_data = cv2image
        self.im_height, self.im_width = cv2image.shape[:2]


class HandModel(FrozenGraph):
    def __init__(self, **kwargs):
        super(HandModel, self).__init__(**kwargs)

    def detect_objects(self):
        self.sess = tf.Session(graph=self.graph)
        with self.sess:
            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
            detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')

            image_np_expanded = np.expand_dims(self.image_data, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

        self.boxes = np.squeeze(boxes)
        self.scores = np.squeeze(scores)
        self.crop_images()

    def crop_images(self):
        self.cropped_hands = []
        for i in range(self.num_hands_detect):
            if self.scores[i] > self.score_thresh:
                box = (self.boxes[i][1] * self.im_width,  self.boxes[i][3] * self.im_width,
                       self.boxes[i][0] * self.im_height, self.boxes[i][2] * self.im_height)
                scale_crop = 0.1
                crop_img = self.image_data[int(box[2]-scale_crop*box[2]): int(box[3]+scale_crop*box[3]),
                                           int(box[0]-scale_crop*box[0]): int(box[1]+scale_crop*box[1])]
                self.cropped_hands.append((crop_img, box, self.scores[i]))


class PoseModel(FrozenGraph):
    def __init__(self, **kwargs):
        super(PoseModel, self).__init__(**kwargs)

    def get_pred(self):
        self.sess = tf.Session(graph=self.graph)
        with self.sess:
            softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
            self.predictions = self.sess.run(softmax_tensor, {'DecodeJpeg:0': self.image_data})
            self.top_k = self.predictions[0].argsort()[-len(self.predictions[0]):][::-1]

        best_score = self.predictions[0][self.top_k[0]]
        name = self.label_lines[self.top_k[0]]
        return name, best_score


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


def draw_box_on_image(box, guess, guess_score, track_score, image_data):
    if track_score > score_thresh:
        (left, right, top, bottom) = (box[0], box[1], box[2], box[3])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))

        if guess is None:
            guess_score = 0
            guess = '???'

        y = top - 5
        label = '{:.1f}%: {}'.format(guess_score * 100, guess)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        cv2.rectangle(image_data, p1, p2, (77, 255, 9), 3, 1)
        size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        cv2.rectangle(image_data, p1, (int(left+size[0]), int(y-5-size[1])), (77, 255, 9), thickness=-1)
        cv2.putText(image_data, label, (int(left), int(y)), font, font_scale, (0, 0, 0), font_thickness)


def draw_fps(image_data):
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time
    cv2.putText(image_data, '{:.2f}'.format(fps), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 77, 255), 2)

if __name__ == '__main__':
    # max number of hands we want to detect/track
    score_thresh = 0.2
    num_hands = 2
    hand_pose_label_lines = ['palm', 'fist', 'point']

    # image_np = cv2.imread('/home/testuser/obj_det_git/tf_image_classifier/multi_test.jpg')

    handTrak = HandModel(score_thresh=score_thresh, num_hands_detect=num_hands)
    handPose = PoseModel(score_thresh=score_thresh, num_hands_detect=num_hands)

    handTrak.load_graph('/home/testuser/obj_det_git/object_tracking/handtracking/hand_inference_graph/frozen_inference_graph.pb')
    handPose.load_graph('/home/testuser/obj_det_git/tf_image_classifier/tf_files/retrained_graph.pb')

    video_capture = WebcamVideoStream(src=0,
                                    width=300,
                                    height=300).start()
    # cap = cv2.VideoCapture('/home/testuser/obj_det_git/myvid.mp4')

    start_time = datetime.datetime.now()
    num_frames = 0

    while True:
        image_np = video_capture.read()
        # ret, image_np = cap.read()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # opencv reads images by default in BGR format
    	if (num_frames % 3) == 0:
	        handTrak.set_input(image_np)
	        handTrak.detect_objects()

        for hand_data in handTrak.cropped_hands:
            if (num_frames % 3) == 0:
                hand_image, box, track_score = hand_data
                handPose.set_input(hand_image)
                guess, guess_score = handPose.get_pred()
            # guess, guess_score = None, None
            draw_box_on_image(box, guess, guess_score, track_score, image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        draw_fps(image_np)
        image_np = cv2.resize(image_np, (0, 0), fx=2, fy=2)
        cv2.imshow('Detection_TF', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            video_capture.stop()
            cv2.destroyAllWindows()
            break