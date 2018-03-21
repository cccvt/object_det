import argparse


def get_args():

    parser = argparse.ArgumentParser()
    # video input
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=320, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=180, help='Height of the frames in the video stream.')
    # detection params
    parser.add_argument('-nhands', '--num_hands', dest='num_hands', type=int,
                        default=2, help='Max number of hands to detect.')
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.2, help='Score threshold for displaying bounding boxes')

    # overlay
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')

    # multiprocessing
    parser.add_argument('-numw', '--num_workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-qsize', '--queue_size', dest='queue_size', type=int,
                        default=1, help='Size of the queue.')


    args = parser.parse_args()
    return args