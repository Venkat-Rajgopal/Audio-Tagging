import numpy as np
import pandas as pd
from classes import GetParameters, DataGenerator
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

#from keras.layers import Convolution1D, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPool1D, Input, MaxPool1D, concatenate
from keras.activations import softmax
from keras import losses, models, optimizers

train_file = pd.read_csv('data/train.csv')
labels = list(train_file['label'].unique())
label_idx = {label: i for i, label in enumerate(labels)} 
train_file.set_index('fname',  inplace=True)
train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])

params = GetParameters(sampling_rate=16000, audio_time=2, n_classes=len(labels), n_folds=10, learning_rate=0.001, max_epochs = 5)

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

def model_1d(params):
    
    nclass = params.n_classes
    input_length = params.audio_length
    
    inp = tf.keras.layers.Input(shape=(input_length,1))

    x = tf.keras.layers.GlobalMaxPool1D()(inp)

    out = tf.keras.layers.Dense(nclass, activation=tf.keras.activations.softmax)(x)

    #model = models.Model(inputs=inp, outputs=out)
    model = tf.keras.Model(inputs = inp, outputs = out)
    opt = tf.keras.optimizers.Adam(params.learning_rate)

    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
    return model

# Run the data generator on the train and test split 
# fit the 1d model 

skf = StratifiedKFold(n_splits=5)
for i, (train_split, val_split) in enumerate(skf.split(train_file.index, train_file.label_idx)):
    train_set = train_file.iloc[train_split]
    val_set = train_file.iloc[val_split]

    print('Training set size:',train_set.shape)

    train_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = train_set.index, labels = train_set['label_idx'], preprocessing_fn=audio_norm)
    val_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = val_set.index, labels = val_set['label_idx'], preprocessing_fn=audio_norm)
    
    model = model_1d(params)
    history = model.fit_generator(train_gen,  validation_data = val_gen, epochs=params.max_epochs, use_multiprocessing=True, workers=6, max_queue_size=20)
    #model.load_weights('best_%d.h5'%i)