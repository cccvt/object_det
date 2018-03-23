import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('/home/testuser/obj_det_git/object_tracking/handtracking2_old/hand_inference_graph/frozen_inference_graph.pb',
                                     '/home/testuser/obj_det_git/object_tracking/handtracking2_old/hand_inference_graph/graph.pbtxt')

# img = cv.imread('/home/testuser/PycharmProjects/obj_detector/images/toy4.jpg')
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, img = cap.read()
    # try:
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # except:
    #     print("Error converting to RGB")

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.8:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    cv.imshow('img', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break