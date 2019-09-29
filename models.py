import keras
from keras.activations import softmax
from keras import losses, models, optimizers

from tensorflow.keras.layers import Input, Convolution2D, Dense, MaxPool2D, BatchNormalization, Activation, Flatten, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

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


def model_conv_2d(params):
    
    nclass = params.n_classes
    
    inp = Input(shape=(params.dim[0],params.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(params.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model