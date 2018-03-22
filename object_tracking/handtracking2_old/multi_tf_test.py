import datetime
import tensorflow as tf
import numpy as np
import cv2
from threading import Thread

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
        self.scores = [0.5 for i in range(self.num_hands_detect)]
        hprc = 0.001*self.im_height
        wprc = 0.001*self.im_width
        self.boxes = [(hprc/self.im_height, wprc/self.im_width, (self.im_height-hprc)/self.im_height, (self.im_width-wprc)/self.im_width) for i in range(self.num_hands_detect)]

class HandDetect(FrozenModel):
    def detect_objects(self):
        self.sess = tf.Session(graph=self.graph)
        with self.sess:
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

class PoseModel(FrozenModel):
    def detect_hand_pose(self):
        for i in range(self.num_hands_detect):
            if (self.scores[i] > self.score_thresh):
                (left, right, top, bottom) = (self.boxes[i][1] * self.im_width, self.boxes[i][3] * self.im_width,
                                              self.boxes[i][0] * self.im_height, self.boxes[i][2] * self.im_height)
                print self.boxes[i]
                print left, right, top, bottom
                scale_selection = 0.05
                left -= scale_selection * left
                right += scale_selection * right
                top -= scale_selection * top
                bottom += scale_selection * bottom

                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))

                # detect hand pose
                self.process_subimg((left, right, top, bottom))
                guess, score = self.classify_hand_pose()
                if guess is None:
                    score = 0
                    guess = '???'

                y = top - 5
                label = '{:.1f}%:{}'.format(score * 100, guess)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1

                cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
                size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                cv2.rectangle(image_np, p1, (int(left + size[0]), int(y - 5 - size[1])), (77, 255, 9), thickness=-1)
                cv2.putText(image_np,
                            label,
                            (int(left), int(y)),
                            font, font_scale, (0, 0, 0), font_thickness)

    def process_subimg(self, box):
        (left, right, top, bottom) = box
        crop_img = self.image_np[int(top): int(bottom),
                   int(left): int(right)]
        crop_img = cv2.resize(crop_img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        self.crop_img = np.expand_dims(crop_img, axis=0)

    def classify_hand_pose(self):
        self.sess = tf.Session(graph=self.graph)
        with self.sess:
            softmax_tensor = self.graph.get_tensor_by_name('final_result:0')

            predictions = self.sess.run(softmax_tensor, {'Mul:0': self.crop_img})
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

if __name__ == '__main__':
    hand_pose_label_lines = ['palm', 'fist', 'point']

    # g = HandDetect(score_thresh=0.2, num_hands=1)
    # g.load_graph('/home/testuser/obj_det_git/object_tracking/handtracking/hand_inference_graph/frozen_inference_graph.pb')

    f = PoseModel(score_thresh=0.2, num_hands=1)
    f.load_graph('/home/testuser/obj_det_git/tf_image_classifier/tf_files/retrained_graph.pb')

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # video_capture = WebcamVideoStream(src=0,
    #                                   width=320,
    #                                   height=240).start()

    start_time = datetime.datetime.now()
    num_frames = 0
    # im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track

    # while True:
    if True:
        # image_np = cv2.imread('/home/testuser/obj_det_git/object_tracking/ocv_vs_tf/test2.png')
        # image_np = cv2.imread('/home/testuser/obj_det_git/t1.jpg')
        image_np = cv2.imread('/home/testuser/obj_det_git/tf_image_classifier/test1.jpg')
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # ret, image_np = cap.read()
        # image_np = video_capture.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # g.set_input(image_np)
        f.set_input(image_np)
        # # actual detection
        # g.detect_objects()
        #
        # f.boxes, f.scores = g.boxes, g.scores
        f.detect_hand_pose()

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        cv2.putText(image_np, '{:.2f}'.format(fps), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 255, 9), 1)

        image_np = cv2.resize(image_np, (0, 0), fx=1, fy=1)
        cv2.imshow('Single Threaded Detection', cv2.cvtColor(
            image_np, cv2.COLOR_RGB2BGR))

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        # cv2.imshow('Detection_TF', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey()
        cv2.destroyAllWindows()
