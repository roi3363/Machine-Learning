import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

rock_dir = os.path.join(os.getcwd(), '../data/rps/rock')
paper_dir = os.path.join(os.getcwd(), '../data/rps/paper')
scissors_dir = os.path.join(os.getcwd(), '../data/rps/scissors')

TRAINING_DIR = os.path.join(os.getcwd(), '../data/rps/')
VALIDATION_DIR = os.path.join(os.getcwd(), '../data/rps-test-set/')

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=126
)
validation_gen = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=126
)
a = next(train_gen)[0][0]
model = tf.keras.Sequential([
    Conv2D(64, 3, activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(train_gen, epochs=20, steps_per_epoch=20, validation_data=validation_gen, verbose=1,
                    validation_steps=3)

model.save('model.h5')