import numpy as np
import pandas as pd
np.random.seed(1)


full_labels = pd.read_csv('data/toy_labels.csv')
full_labels.head()
grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

# split each file into a group in a list
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]

train_index = np.random.choice(len(grouped_list), size=60, replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)

# take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)
