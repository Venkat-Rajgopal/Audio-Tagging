import numpy as np
import pandas as pd
import os
from classes import GetParameters, DataGenerator
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from models import model_1d, model_conv_2d

print('Project directory:', os.getcwd())

train_file = pd.read_csv('data/train.csv')
labels = list(train_file['label'].unique())
label_idx = {label: i for i, label in enumerate(labels)} 
train_file.set_index('fname',  inplace=True)
train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])

params = GetParameters(sampling_rate=16000, audio_time=2, n_classes=len(labels), n_folds=10, learning_rate=0.001, max_epochs = 50)

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

# Run the data generator on the train and test split 
# fit the 1d model 

skf = StratifiedKFold(n_splits=5)
for i, (train_split, val_split) in enumerate(skf.split(train_file.index, train_file.label_idx)):
    train_set = train_file.iloc[train_split]
    val_set = train_file.iloc[val_split]
    
    print("Fold: ", i)
    print('Training set size:',train_set.shape)

    train_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = train_set.index, labels = train_set['label_idx'], preprocessing_fn=audio_norm)
    val_gen = DataGenerator(data_dir = 'data/audio_train/', parameters = params, list_IDs = val_set.index, labels = val_set['label_idx'], preprocessing_fn=audio_norm)
    
    # save model weights to the same file iff val accuracy inproves

    checkpoint = ModelCheckpoint('weights.best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True, mode = 'min')
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    callbacks_list = [checkpoint, early_stop]

    model = model_conv_2d(params)
    model_fit = model.fit_generator(train_gen,  validation_data = val_gen, callbacks = callbacks_list, epochs=params.max_epochs, use_multiprocessing=False, workers=6, max_queue_size=20)
    
    model.load_weights('weights.best_%d.h5'%i)

    # get predictions
    predictions = model.predict_generator(val_gen, use_multiprocessing=True, workers=6, max_queue_size=20, verbose=1)
    print(predictions)