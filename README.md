# Object detection using Tensorflow and OpenCV

* [hand_pose_dataset](https://github.com/parvudan/object_det/tree/master/hand_pose_dataset) 
  - contains `fist`, `palm` and `point` training images
* [image_classifier](https://github.com/parvudan/object_det/tree/master/image_classifier) 
  - is an attempt at using a convolutional neural net to classify images of hands
* [object_tracking](https://github.com/parvudan/object_det/tree/master/object_tracking) 
  - attempts to use nn to track hands, mostly based on ssd_mobilenet_v1_coco_2017_11_17
  - `*.config` file obtained from https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
  - handtracking code from https://github.com/victordibia/handtracking
* [opencv_dnn](https://github.com/parvudan/object_det/tree/master/opencv_dnn)
  - quick attempt at loading a tensorflow net in opencv using `cv2.readNetFromTensorflow` (not working/ wrong predictions)
* [tf_image_classifier](https://github.com/parvudan/object_det/tree/master/tf_image_classifier) 
  - image classification using the inception-2015-12-05 model 
  - cloned from https://github.com/burliEnterprises/tensorflow-image-classifier.git
