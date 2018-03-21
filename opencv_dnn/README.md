The OpenCV readNetFromTensorflow command takes two arguments:
1. the frozen inference graph (*.pb), which is a binary tensorflow model
2. the textual representation of this model, which can be generated using [tf_text_graph_ssd.py](https://github.com/opencv/opencv/blob/master/samples/dnn/tf_text_graph_ssd.py),
as described here: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#generate-a-config-file

Issues:
1. if OpenCV raises an error like:

`OpenCV Error: Assertion failed (ngroups > 0 && inpCn % ngroups == 0 && outCn % ngroups == 0) in getMemoryShapes, file /io/opencv/modules/dnn/src/layers/convolution_layer.cpp, line 217`

the input image has the wrong number of channels, i.e. the model expects that the image has a specific number of channels. Try to enable/disable greyscale mode when reading in the image
