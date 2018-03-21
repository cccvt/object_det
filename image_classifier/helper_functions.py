import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import tensorflow as tf
import random
# from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from settings import IMG_SIZE, TRAIN_DIR, TEST_DIR, CLASS_LABELS

def label_img(img):
    word_label = img.split('-')[0]
    # conversion to one-hot array [A, B, C, Five, Point, V]
    valid_labels = CLASS_LABELS
    out = [0 for i in valid_labels]
    if word_label in valid_labels:
        x = valid_labels.index(word_label)
        out[x] = 1
    return out


def make_data(data_dir):
    data = []
    for img in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, img)):
            data.extend(make_data(os.path.join(data_dir, img)))
        else:
            label = label_img(img)
            path  = os.path.join(data_dir, img)
            img   = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append([np.array(img), np.array(label)])
    return data


def save_data(data, shuffle_data=False, save_path=None):
    # if shuffle_data:
    #     shuffle(data)
    if save_path:
        np.save(save_path, data)


def make_train_data():
    try:
        # train_data = np.load('train_data.npy')
        # TODO: fix this
        1/0
    except:
        train_data = make_data(TRAIN_DIR)
        train_data = random.sample(train_data, len(train_data))
        save_data(train_data, shuffle_data=True, save_path='train_data.npy')
    return train_data


def make_test_data():
    try:
        # test_data = np.load('test_data.npy')
        # TODO: fix this also
        1/0
    except:
        test_data = make_data(TEST_DIR)
        test_data = random.sample(test_data, len(test_data))
        save_data(test_data, shuffle_data=True, save_path='test_data.npy')
    return test_data