import numpy as np
import matplotlib.pyplot as plt
from helper_functions import make_test_data, make_train_data
from settings import IMG_SIZE, CLASS_LABELS


def plot_results(model):
    test_data = make_test_data()

    fig = plt.figure()

    for num, data in enumerate(test_data[:12]):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        actual_label = CLASS_LABELS[np.argmax(img_num)]
        guess_label = CLASS_LABELS[np.argmax(model_out)]

        y.imshow(orig, cmap='gray')
        plt.title('pred:{}, actual:{}'.format(guess_label, actual_label))
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()