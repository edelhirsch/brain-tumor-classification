import glob
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os
import shutil
import tensorflow as tf

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
    ### (1) try increasing image size even more (256 is too high)
    ### set this to 128 to make it go faster:
    size = (150, 150)
    input_shape = size + (3,)
    epochs = 20
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(original_training_dir,
                                                                   label_mode='categorical',
                                                                   color_mode='rgb',  # or rgb?
                                                                   image_size=size,
                                                                   batch_size=batch_size,
                                                                   validation_split=0.2,
                                                                   subset='training',
                                                                   seed=1337,
                                                                   )
    print(f'classes found: {train_ds.class_names}')
    print(f'train ds: {train_ds}')
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(original_training_dir,
                                                                        label_mode='categorical',
                                                                        color_mode='rgb',
                                                                        image_size=size,
                                                                        batch_size=batch_size,
                                                                        validation_split=0.2,
                                                                        subset='validation',
                                                                        seed=1337,
                                                                        )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(original_testing_dir,
                                                                  label_mode='categorical',
                                                                  color_mode='rgb',
                                                                  image_size=size,
                                                                  batch_size=batch_size,
                                                                  )

    # train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
        ]
    )

    # for images, labels in train_ds.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = images[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(
    #             tf.expand_dims(first_image, 0), training=True
    #         )
    #         plt.imshow(augmented_image[0].numpy().astype("int32"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")

    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(4)(x)
    model = keras.Model(inputs, outputs)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    model.save('brain-tumor-classification.h5')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    build_model()

