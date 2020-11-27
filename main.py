#!/usr/bin/python3

import argparse
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.xception import decode_predictions

base_dir = "/home/peter/dev/brain-tumors"
original_data_dir = base_dir + "/original-data"
original_training_dir = original_data_dir + "/Training"
original_testing_dir = original_data_dir + "/Testing"
inference_dir = base_dir + "/inference"

### We don't need these:
data_dir = base_dir + "/data"  # use os.path.join
training_dir = data_dir + "/Training"
validation_dir = data_dir + "/Validation"
testing_dir = data_dir + "/Testing"

model_name = 'brain-tumor-classification.h5'

size = (150, 150)
input_shape = size + (3,)
epochs = 20
batch_size = 32


def create_datasets():

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(original_training_dir,
                                                                   label_mode='categorical',
                                                                   color_mode='rgb',  # or rgb?
                                                                   image_size=size,
                                                                   batch_size=batch_size,
                                                                   validation_split=0.2,
                                                                   subset='training',
                                                                   seed=1337,
                                                                   )

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
    return train_ds, validation_ds, test_ds


def show_images():
    print('*** Showing images ***')
    train_ds, validation_ds, test_ds = create_datasets()
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        first_image = images[0]
        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().astype("int32"))
            index = int(tf.math.argmax(labels[i]))
            plt.title(train_ds.class_names[index])
            plt.axis("off")
    plt.show()


def build_model():
    print('*** Building the model ***')
    train_ds, validation_ds, test_ds = create_datasets()
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomFlip("vertical"),
            layers.experimental.preprocessing.RandomWidth(0.2),
            layers.experimental.preprocessing.RandomHeight(0.2),
            layers.experimental.preprocessing.RandomContrast(0.2),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
            layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
        ]
    )

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
        optimizer=keras.optimizers.Adam(0.00025),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    model.save(model_name)

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def test_model():
    print('*** Testing the model ***')
    ### check whether model does exist
    model = keras.models.load_model(model_name)

    train_ds, validation_ds, test_ds = create_datasets()
    evaluation = model.evaluate(test_ds)
    print(evaluation)


def predict():
    print('*** Predicting new data ***')
    model = keras.models.load_model(model_name)
    image_path = '/home/peter/dev/brain-tumors/inference/pituitary_tumor.jpg'
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
    x = keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x).flatten()
    #decode_predictions(predictions)

    train_ds, validation_ds, test_ds = create_datasets()

    for index, value in zip(train_ds.class_names, predictions):
        print(f'{index}: {value}')

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    index = np.argmax(predictions)
    title = os.path.basename(image_path) + "\nprediction: " + train_ds.class_names[index]
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run machine learning for brain tumor classification.')
    parser.add_argument('--show-images', action='store_true', default=False, help='show the first 10 images')
    parser.add_argument('--build-model', action='store_true', default=True, help='builds the model')
    parser.add_argument('--test-model', action='store_true', default=False, help='tests the model')
    parser.add_argument('--predict', action='store_true', default=False, help='predicts the model for other files')
    args = parser.parse_args()

    if args.show_images:
        show_images()
    elif args.test_model:
        test_model()
    elif args.predict:
        predict()
    elif args.build_model:
        build_model()
