import os
import numpy as np
from plot_results import plot_results
from setup_nn_layers import make_nn_layers
from helper_functions import make_train_data
from settings import MODEL_NAME, IMG_SIZE, TRAIN_MODEL, EPOCHS


def train_network(show_results=True):
    model = make_nn_layers()

    model_loaded = False
    train_flag = False

    if os.path.exists('{}.meta'.format(os.path.join('models', MODEL_NAME))):
        model.load(os.path.join('models', MODEL_NAME))
        model_loaded = True
        print('Model checkpoint loaded!')

    if TRAIN_MODEL:
        train_flag = True

    if TRAIN_MODEL and not model_loaded:
        train_flag = True
        print 'No model loaded!\nStart training...'

    if not TRAIN_MODEL and not model_loaded:
        raise Exception('No model loaded!\nTRAIN_MODEL set to False!\nNothing to run!')

    if train_flag:
        # train the model
        train_data = make_train_data()

        split = int(0.8*len(train_data))
        train = train_data[:split]
        test = train_data[split:]

        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        Y = np.array([i[1] for i in train])

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        test_y = [i[1] for i in test]

        model.fit({'input': X},
                  {'targets': Y},
                  n_epoch=EPOCHS,
                  validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500,
                  show_metric=True,
                  run_id=MODEL_NAME)

        model.save(os.path.join('models', MODEL_NAME))

    if show_results:
        plot_results(model)
