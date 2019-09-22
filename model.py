import numpy as np
import pandas as pd
from classes import GetParameters, DataGenerator
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from keras.layers import Convolution1D, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPool1D, Input, MaxPool1D, concatenate

train_file = pd.read_csv('data/train.csv')
labels = list(train_file['label'].unique())
label_idx = {label: i for i, label in enumerate(labels)} 
train_file.set_index('fname',  inplace=True)
train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])


skf = StratifiedKFold(n_splits=5)

params = GetParameters(sampling_rate=16000, audio_time=2, n_classes=41, n_folds=5, learning_rate=0.001, max_epochs = 50)

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

def dummy_model(params):
    
    nclass = params.n_classes
    input_length = params.audio_duration
    
    inp = Input(shape=(input_length,1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(params.learning_rate)

    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['acc'])
    return model

for i, (train_split, val_split) in enumerate(skf.split(train_file.index, train_file.label_idx)):
    train_set = train_file.iloc[train_split]
    val_set = train_file.iloc[val_split]

    train_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = train_set.index, labels = train_set['label_idx'])
    val_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = val_set.index, labels = val_set['label_idx'])
    
    model = dummy_model(params)