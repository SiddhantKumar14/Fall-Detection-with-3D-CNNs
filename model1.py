from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import GlobalMaxPool2D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Reshape, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils, generic_utils
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import numpy as np
from sklearn import preprocessing

nb_classes = 3

print('Building model: ', end="")
model = Sequential()
#`channels_last` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
strides = (1,1,1)
kernel_size = (3, 3, 3)

model.add(Conv3D(32, kernel_size, strides=strides, activation='relu', padding='same', input_shape=(30, 64, 96, 3)))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(0.25))

model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(0.25))

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(MaxPooling3D(pool_size=(1,8,12)))
print(model.output_shape)

model.add(Reshape((30, 256)))
print(model.output_shape)
model.add(LSTM(256, return_sequences=True))
print(model.output_shape)
model.add(LSTM(256))
print(model.output_shape)

model.add(Dense(256, activation='relu'))
print(model.output_shape)
#model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))
print(model.output_shape)
print('done')