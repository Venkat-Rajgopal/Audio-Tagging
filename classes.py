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
        self.dim = (self.audio_length, 1)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dir, parameters, list_IDs, labels, batch_size=32, shuffle=True):
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
        return self.__data_generation(list_IDs_temp)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch size samples'
        print(list_IDs_temp)

        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        input_length = self.GetParameters.audio_duration
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID

            data, _ = librosa.core.load(file_path, sr=self.GetParameters.sampling_rate, res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype = int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
                return X, keras.utils.to_categorical(y, num_classes=self.GetParameters.n_classes)
            else:
                return X








