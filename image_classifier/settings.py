from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M")


# define input data
TRAIN_DIR = '/home/testuser/tf_tut/hand_pose_datasets/selected'
TEST_DIR = '/home/testuser/tf_tut/hand_pose_datasets/selected'
IMG_SIZE = 64  # resize images to this number

TRAIN_MODEL = True  # continue trainig (if False, start training)
LR = 0.0005  # learning rate
EPOCHS = 20  # foe how many epochs to train


# define classes
CLASS_LABELS = ['fist', 'palm', 'point']
NUM_CLASSES = len(CLASS_LABELS)

# define model name
# MODEL_NAME = 'handgesture-{}-{}-{}.model'.format(LR, timestamp, '5conv-basic') # just so we remember which saved model is which, sizes must match
MODEL_NAME = 'handgesture-0.0005-20180319-1358-5conv-basic.model'