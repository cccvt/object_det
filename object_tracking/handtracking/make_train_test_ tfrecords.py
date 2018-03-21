import os

pathlist = [('data/train_labels.csv', 'data/train.record'),
            ('data/test_labels.csv', 'data/test.record')]

for csv_in, tf_out in pathlist:
    os.system('python generate_tfrecord.py --csv_input={0}  --output_path={1}'.format(csv_in, tf_out))