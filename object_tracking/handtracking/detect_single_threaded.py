from utils import detector_utils as detector_utils
from utils.handle_args import get_args
import cv2
import tensorflow as tf
import datetime
import argparse

# cap = cv2.VideoCapture(0)
# video from screen cap:
# sudo modprobe v4l2loopback
# ffmpeg -f x11grab -r 15 -s 640x480 -i :0.0+0,0 -vcodec rawvideo -pix_fmt rgb24 -threads 0 -f v4l2 /dev/video0

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    args = get_args()

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
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # actual detection
        boxes, scores = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        # draw bounding boxes
        detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

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

            cv2.imshow('Single Threaded Detection', cv2.cvtColor(
                image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
