# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:52:40 2019

@author: Peter Wang
"""

from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD,Adam

import numpy as np
import pandas as pd
import random

google_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

model = Sequential()  
model.add(Conv2D(8, (4, 4), strides = 3, input_shape = (100, 100, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(16, (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Third convolutional layer
model.add(Conv2D(32, (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(28))
model.add(Activation('relu'))
model.add(Dense(28))
model.add(Activation('softmax'))

model.compile(loss='mse',optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'])
model.summary()

model.fit_generator(generator=google_gen.flow_from_directory('data/Messier_npy_png_train', target_size=(100, 100), color_mode='rgb'), 
                    validation_data=google_gen.flow_from_directory('data/Messier_npy_png_test', target_size=(100, 100), color_mode='rgb'), 
                    steps_per_epoch=3000,
                    validation_steps=300, 
                    epochs = 20
                    )