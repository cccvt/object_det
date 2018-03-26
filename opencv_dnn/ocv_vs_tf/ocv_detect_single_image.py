import cv2


if __name__ == '__main__':
#    image_np = cv2.imread('/home/testuser/obj_det_git/opencv_dnn/ocv_vs_tf/test2.png')
    image_np = cv2.imread('/home/testuser/obj_det_git/image_classifier/t1.jpg')

    mean = 127.5
    blob = cv2.dnn.blobFromImage(image_np, 1/mean, (300, 300), (mean, mean, mean), swapRB=True, crop=False)
    net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

    net.setInput(blob)
    out = net.forward()

    rows = image_np.shape[0]
    cols = image_np.shape[1]
    for detection in out[0,0,:,:]:
        score = float(detection[2])
        if score > 0.2:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), 3, 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            cv2.putText(image_np,
                    '{:.2f}'.format(score),
                    (int(left), int(top+5)),
                    font, font_scale, (0, 0, 255), font_thickness)

    cv2.imshow('Detection_OCV', image_np)
    cv2.imwrite('opencv_result2.png', image_np)
    cv2.waitKey()
