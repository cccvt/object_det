import cv2 as cv
import datetime

# cvNet = cv.dnn.readNetFromTensorflow('/home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/frozen_inference_graph.pb',
#                                      '/home/testuser/PycharmProjects/obj_detector/handtracking/hand_inference_graph/graph.pbtxt')
cvNet = cv.dnn.readNetFromTensorflow('/home/testuser/obj_det_git/object_tracking/handtracking2/hand_inference_graph/frozen_inference_graph.pb',
                                     '/home/testuser/obj_det_git/object_tracking/handtracking2/hand_inference_graph/fixed_graph.pbtxt')

# img = cv.imread('/home/testuser/PycharmProjects/obj_detector/images/toy14.jpg')
cap = cv.VideoCapture(0)
start_time = datetime.datetime.now()
num_frames = 0

while True:
    ret, img = cap.read()

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.6:
            print (detection[3], detection[5], detection[4], detection[6])
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            # box = (left, top, right, bottom)
            # print box
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    num_frames += 1
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time

    cv.putText(img, '{:.1f}'.format(fps), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (77, 255, 9), 2)
    cv.imshow('img', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
