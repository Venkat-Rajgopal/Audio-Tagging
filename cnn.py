import librosa
import numpy as np
import pandas as pd
import scipy

# Model imports
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical


# ------------------------------------------------------------------------------------
# Configure parameters
sampling_rate = 16000
audio_duration = 2
n_classes = 41
n_folds = 10
learning_rate = 0.0001
max_epochs = 50
audio_length = sampling_rate * audio_duration

# mfcc parameters here 

# ------------------------------------------------------------------------------------
# Generate Data 
preprocessing_fn = lambda x: x 

# Prepare label indices
train_file = pd.read_csv('data/train.csv')
labels = list(train_file.label.unique())

# index each label item 
label_idx = {label: i for i, label in enumerate(labels)} 

train_file.set_index('fname',  inplace=True)
#test.set_index("fname", inplace=True)

train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])




# ------------------------------------------------------------------------------------
def min_max_norm(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
    return data - 0.5


def 1d_test_model(n_classes, audio_length, learning_rate):

    """ This is a test model for checking 
    Input(): initiates a Keras Tensor
    """

    input_length = audio_length
    inp = Input(shape= input_length, 1)

    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs = inp, outputs = out)
    opt = optimizers.Adam(learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model



