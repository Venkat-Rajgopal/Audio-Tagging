import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Prepare Partitions dictionary
train_file = pd.read_csv('data/train.csv')
test_file = pd.read_csv('data/test_post_competition.csv')

# index each unique label item 
labels = list(train_file['label'].unique())
label_idx = {label: i for i, label in enumerate(labels)} 

# assign the index to the training file and update a new column. 
train_file.set_index('fname',  inplace=False)
#test.set_index("fname", inplace=True)

train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])


# train_file.loc['0048fd00.wav']

fnames =  list(train_file['fname'])

# Random shuffling
indexes = np.arange(len(fnames))
train_file['ID'] = indexes

#np.random.shuffle(indexes)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(train_file['fname'],  train_file['label'])
partition = {}

for train_index, val_index in skf.split(train_file['fname'],  train_file['label']):
    partition['train'], partition['validation'] = train_file['fname'][train_index], train_file['fname'][val_index]

print(len(partition['train']))
print(len(partition['validation']))