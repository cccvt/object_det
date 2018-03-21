from settings import MODEL_NAME, IMG_SIZE, CLASS_LABELS
from setup_nn_layers import make_nn_layers
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = make_nn_layers()
model.load(MODEL_NAME)

img_data   = cv2.imread('/home/testuser/tf_tut/image_classifier/Marcel-Test/A/complex/A-complex07.ppm', cv2.IMREAD_GRAYSCALE)
img_data   = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
model_out = model.predict([data])[0]

p = plt.imshow(img_data, cmap='gray')
plt.title('pred: {}'.format(CLASS_LABELS[np.argmax(model_out)]))
p.axes.get_xaxis().set_visible(False)
p.axes.get_yaxis().set_visible(False)
plt.show()