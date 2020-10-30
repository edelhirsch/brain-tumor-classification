# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import glob
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from os import path
import os
import shutil

base_dir = "/home/peter/dev/brain-tumors"
original_data_dir = base_dir + "/original-data"
original_training_dir = original_data_dir + "/Training"
original_testing_dir = original_data_dir + "/Testing"

data_dir = base_dir + "/data"  # use os.path.join
training_dir = data_dir + "/Training"
validation_dir = data_dir + "/Validation"
testing_dir = data_dir + "/Testing"

categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(),
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    if not path.exists(training_dir):
        os.mkdir(data_dir)
        os.mkdir(training_dir)

    train_generator = train_datagen.flow_from_directory(
        original_training_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        save_to_dir=training_dir,
        save_prefix='',
        save_format='jpeg',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        original_training_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        subset='validation'
    )

    history = model.fit(
        x=train_generator,
        epochs=30,
        verbose=2,
        validation_data=validation_generator,
    )

    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=100,
    #     Batch_size=2,
    #     epochs=30,
    #     validation_data=validation_generator,
    #     validation_steps=50)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    build_model()

