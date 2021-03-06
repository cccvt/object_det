# Object detection using Tensorflow and OpenCV

* [hand_pose_dataset](https://github.com/parvudan/object_det/tree/master/hand_pose_dataset) 
  - contains `fist`, `palm` and `point` training images
* [object_tracking](https://github.com/parvudan/object_det/tree/master/object_tracking) 
  - attempts to use nn to track hands, mostly based on ssd_mobilenet_v1_coco_2017_11_17
  - `*.config` file obtained from https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
  - handtracking code from https://github.com/victordibia/handtracking
* [opencv_dnn](https://github.com/parvudan/object_det/tree/master/opencv_dnn)
  - quick attempt at loading a tensorflow net in opencv using `cv2.readNetFromTensorflow` (not working/ wrong predictions)
* [tensorflow-for-poets-2](https://github.com/parvudan/object_det/tree/master/tensorflow-for-poets-2) 
  - retrained model with own image dataset to detect palm, fist or point
  - repo contains code for the "TensorFlow for poets 2" series of codelabs
  - [Tutorial Part 1](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)
  - [Tutorial Part 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite)
