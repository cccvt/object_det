import numpy as np
import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('/home/testuser/object_det/DeepLearningExample/hat_model/frozen_inference_graph.pb', 
									 '/home/testuser/object_det/DeepLearningExample/hat_model/test.pbtxt')

img = cv.imread('images/image5.jpg')
cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.5:
        left = detection[3] * img.shape[1]
        top = detection[4] * img.shape[0]
        right = detection[5] * img.shape[1]
        bottom = detection[6] * img.shape[0]
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0))

cv.imshow('img', img)
cv.waitKey()
