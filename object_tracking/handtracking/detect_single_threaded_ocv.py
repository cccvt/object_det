from utils import detector_utils as detector_utils
import numpy as np
import cv2
import tensorflow as tf
import datetime
import argparse

# cap = cv2.VideoCapture(0)
# detection_graph, sess = detector_utils.load_inference_graph()
# cvNet = cv2.dnn.readNetFromTensorflow('/home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/frozen_inference_graph.pb',
#                                       '/home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/graph.pbtxt')
cvNet = cv2.dnn.readNetFromTensorflow('/home/testuser/PycharmProjects/obj_detector/DeepLearningObjectDetection/hat_model/frozen_inference_graph.pb',
                                      '/home/testuser/PycharmProjects/obj_detector/DeepLearningObjectDetection/hat_model/ssd_mobilenet_v1_coco.pbtxt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.9, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        # try:
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # except:
        #     print("Error converting to RGB")

        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
        cvNet.setInput(blob)
        detections = cvNet.forward()

        (h, w) = image_np.shape[:2]
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args.score_thresh:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image_np, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image_np, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # boxes = []
        # scores = []
        # for detection in cvOut[0, 0, :, :]:
        #     left = detection[3]
        #     top = detection[4]
        #     right = detection[5]
        #     bottom = detection[6]
        #     box = (left, right, top, bottom)
        #     boxes.append(box)
        #     scores.append(float(detection[2]))
        #
        # for i in range(num_hands_detect):
        #     if scores[i] > args.score_thresh:
        #         (left, right, top, bottom) = (boxes[i][0] * im_width, boxes[i][1] * im_width,
        #                                       boxes[i][2] * im_height, boxes[i][3] * im_height)
        #         p1 = (int(left), int(top))
        #         p2 = (int(right), int(bottom))
        #         cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            # score = float(detection[2])
            # if score > 0.8:
                # left = detection[3] * cols
                # top = detection[4] * rows
                # right = detection[5] * cols
                # bottom = detection[6] * rows
                # cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

        # actual detection
        # boxes, scores = detector_utils.detect_objects(
        #     image_np, detection_graph, sess)

        # draw bounding boxes
        # detector_utils.draw_box_on_image(
        #     num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), image_np)

            # cv2.imshow('Single Threaded Detection', cv2.cvtColor(
            #     image_np, cv2.COLOR_RGB2BGR))
            cv2.imshow('Single Threaded Detection', image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
