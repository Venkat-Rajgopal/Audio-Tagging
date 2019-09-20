import librosa
import numpy as np
import scipy
import keras 


class GetParameters(object):
    'Class containing all HyperParameters'
    def __init__(self, sampling_rate, audio_time, n_classes, n_folds, learning_rate, max_epochs):

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_time
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, parameters, list_IDs, labels, batch_size=32, shuffle=True):
        'Initialization'
        self.parameters = parameters
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # generates indexes for the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # find list of ID's
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch size samples'
        print(list_IDs_temp)



params = GetParameters(sampling_rate=16000, audio_time=2, n_classes=41, n_folds=5, learning_rate=0.001, max_epochs = 50)









