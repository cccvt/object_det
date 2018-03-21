import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from settings import IMG_SIZE, LR, NUM_CLASSES


def make_nn_layers():
    # define neural network layers
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, NUM_CLASSES, activation='softmax')

    acc = Accuracy(name="Accuracy")
    convnet = regression(convnet, optimizer='adam',
                         learning_rate=LR,
                         loss='categorical_crossentropy',
                         name='targets',
                         metric=acc)

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model