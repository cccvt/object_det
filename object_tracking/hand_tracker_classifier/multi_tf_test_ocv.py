import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from multiprocessing import Queue, Pool
from utils import WebcamVideoStream, HandModelCV, draw_box_on_image, draw_fps, worker_hand_pose

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow warnings

if __name__ == '__main__':
    # max number of hands we want to detect/track
    score_thresh = 0.2
    num_hands = 2

    handTrak = HandModelCV(score_thresh=0.6, num_hands_detect=num_hands)
    # handPose = PoseModel(score_thresh=score_thresh, num_hands_detect=num_hands)

    handTrak.load_graph('frozen_models/hand_detect_graph.pb',
                        'frozen_models/hand_detect_graph.pbtxt')
    # handPose.load_graph('/home/testuser/obj_det_git/tf_image_classifier/tf_files/retrained_graph.pb')

    video_capture = WebcamVideoStream(src=0,
                                      width=800,
                                      height=600).start()
    num_frames = 0
    start_time = datetime.datetime.now()

    # init multiprocessing
    input_q = Queue(maxsize=5)
    output_q = Queue(maxsize=5)

    # spin up workers to paralleize detection.
    frame_processed = 0
    num_workers = 3
    pool_hand = Pool(num_workers, worker_hand_pose, (input_q, output_q, frame_processed))

    while True:
        image_np = video_capture.read()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # opencv reads images by default in BGR format
        handTrak.set_input(image_np)
        handTrak.detect_objects()

        for hand_data in handTrak.cropped_hands:
            hand_image, box, track_score = hand_data
            if hand_image.shape[0] > 0 and hand_image.shape[1] > 0:
                input_q.put(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))
                _, guess, guess_score = output_q.get()
                print guess, guess_score
                # handPose.set_input(hand_image)
                # guess, guess_score = handPose.get_pred()
                draw_box_on_image(box, guess, guess_score, track_score, score_thresh, image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        draw_fps(start_time, num_frames, image_np)
        image_np = cv2.resize(image_np, (0, 0), fx=1, fy=1)
        cv2.imshow('Detection_TF', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            video_capture.stop()
            cv2.destroyAllWindows()
            break
