import numpy as np
import pandas as pd


np.random.seed(1)


full_labels = pd.read_csv('data/object_labels.csv')
grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

# split each file into a group in a list
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]

# use 80% of data as training data, rest as test data
train_index = np.random.choice(len(grouped_list), size=int(0.8*len(grouped_list)), replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)

train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)

