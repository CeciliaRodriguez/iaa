from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 150
IMG_WIDTH = 150
batch_size = 128


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.load_weights('model.h5')

test_images_gen = ImageDataGenerator(rescale=1./255)
test_images_data = test_images_gen.flow_from_directory(batch_size=batch_size,
                                                 directory=r'C:\Users\CRODRIGU\.keras\datasets\cats_and_dogs_filtered\test',
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH))

predictions = model.predict(test_images_data)

print(predictions)
