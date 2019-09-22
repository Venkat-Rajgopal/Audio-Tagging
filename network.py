import librosa
import numpy as np
import pandas as pd
import scipy
import keras 
from sklearn.model_selection import StratifiedKFold

# Set all parameters 
sampling_rate = 16000
audio_time = 2
n_classes = 41
n_folds = 5
learning_rate=0.001
max_epochs = 50
dim = (audio_time, 1)
data_dir = 'data/audio_train/'
batch_size = 32
# Make data generation functions here


def n_batches():
    return int(np.floor(len(list_IDs) / batch_size))


def on_epoch_end():
    indexes = np.arange(len(list_IDs))
    return indexes

def get_item(indexes, index):
    indexes = indexes[index*batch_size:(index+1)*batch_size]

    list_IDs_temp = [list_IDs[k] for k in indexes]

    return data_generation(list_IDs_temp)

def data_generation(list_IDs_temp):
    cur_batch_size = len(list_IDs_temp)
    X = np.empty((cur_batch_size, dim))


    input_length = audio_time

    for i, ID in enumerate(list_IDs_temp):
        print(i)
        print(ID)



# -------------------------------------------------------------------------------
train_file = pd.read_csv('data/train.csv')
labels = list(train_file['label'].unique())
label_idx = {label: i for i, label in enumerate(labels)} 
train_file.set_index('fname',  inplace=True)
train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])


skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(train_file['fname'],  train_file['label_idx'])

for train_index, val_index in skf.split(train_file['fname'],  train_file['label_idx']):
    train_set = train_file.iloc[train_index]
    val_set = train_file.iloc[val_index]

    list_IDs = train_set.index
    labels = train_set['label_idx']

    indexes = on_epoch_end()

    get_item(indexes=indexes, index = np.array(train_set.index).astype(int))


for i, (train_split, val_split) in enumerate(skf.split(train_file.index, train_file.label_idx)):
    train_set = train_file.iloc[train_split]
    val_set = train_file.iloc[val_split]

    list_IDs = train_set.index
    labels = train_set['label_idx']

    indexes = on_epoch_end()
    get_item(indexes=indexes, index = 1)

