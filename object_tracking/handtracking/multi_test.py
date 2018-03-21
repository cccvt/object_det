import cv2
import tensorflow as tf
import multiprocessing as mp
import datetime
from utils import detector_utils
from utils.handle_args import get_args
from utils.detector_utils import WebcamVideoStream


# def run_session(tup):
#     print(">> loading frozen model for worker")
#     frame, cap_params, path = tup
#     detection_graph, sess = detector_utils.load_inference_graph(path)
#     sess = tf.Session(graph=detection_graph)
#     print("> ===== in worker loop, frame ")
#     if (frame is not None):
#         # actual detection
#         boxes, scores = detector_utils.detect_objects(
#             frame, detection_graph, sess)
#         # draw bounding boxes
#         detector_utils.draw_box_on_image(
#             cap_params['num_hands_detect'],
#             cap_params['score_thresh'],
#             scores,
#             boxes,
#             cap_params['im_width'],
#             cap_params['im_height'],
#             frame)
#     sess.close()
#     return frame

def run_session(tup):
    import tensorflow as tf

    frame, cap_params, path = tup
    # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 2}, log_device_placement=False))
    print("> ===== in worker loop, frame ")
    with tf.device("/cpu:0"):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with tf.Session(graph=detection_graph) as sess:
            if (frame is not None):
                # actual detection
                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)
                # draw bounding boxes
                detector_utils.draw_box_on_image(
                    cap_params['num_hands_detect'],
                    cap_params['score_thresh'],
                    scores,
                    boxes,
                    cap_params['im_width'],
                    cap_params['im_height'],
                    frame)

frame_processed = 0
score_thresh = 0.2

if __name__ == '__main__':
    args = get_args()

    # video_capture = WebcamVideoStream(src=args.video_source,
    #                                   width=args.width,
    #                                   height=args.height).start()
    cap = cv2.VideoCapture(args.video_source)
    im_width, im_height = (cap.get(3), cap.get(4))

    cap_params = {}
    frame_processed = 0
    # cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['im_width'], cap_params['im_height'] = (im_width, im_height)
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands
    start_time = datetime.datetime.now()
    num_frames = 0
    while True:
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)

        inputs = (cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), cap_params,
                  '/home/testuser/tf_tut/obj_detection/handtracking/hand_inference_graph/frozen_inference_graph.pb')
        # run_session(inputs)
        p = mp.Pool(2)
        p.map(run_session, (inputs,))
        p.close()
        p.join()

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
