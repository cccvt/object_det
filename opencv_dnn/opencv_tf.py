import cv2 as cv
import numpy as np

frame = cv.imread('/home/testuser/obj_det_git/image_classifier/t1.jpg')
rows = frame.shape[0]
cols = frame.shape[1]

net = cv.dnn.readNetFromTensorflow('/home/testuser/obj_det_git/object_tracking/handtracking2_old/hand_inference_graph/frozen_inference_graph.pb',
                                   '/home/testuser/obj_det_git/object_tracking/handtracking2_old/hand_inference_graph/graph.pbtxt')

classes = None

blob = cv.dnn.blobFromImage(frame, 1, (299, 299), (127.5, 127.5, 127.5), swapRB=True, crop=False)

net.setInput(blob)

out = net.forward()

for detection in out[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', frame)
cv.waitKey()


# # opencv dnn inference
# # Get a class with a highest score.
# out = out.flatten()
# classId = np.argmax(out)
# confidence = out[classId]
#
# # Put efficiency information.
# t, _ = net.getPerfProfile()
# label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
# cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
# # Print predicted class.
# label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
# cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
# cv.imshow('out', frame)