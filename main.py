import cv2
import glob
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import numpy as np
from os import path
import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "/home/peter/dev/brain-tumors"
original_data_dir = base_dir + "/original-data"
original_training_dir = original_data_dir + "/Training"
original_testing_dir = original_data_dir + "/Testing"

data_dir = base_dir + "/data"  # use os.path.join
training_dir = data_dir + "/Training"
validation_dir = data_dir + "/Validation"
testing_dir = data_dir + "/Testing"

categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
img_size = 150



def create_training_data():
    training_data = []
    x = []
    y = []

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    os.mkdir(data_dir)
    os.mkdir(training_dir)
    ### create testing data

    for category in categories:
        print(f'creating training data for {category}')
        new_dir = os.path.join(training_dir, category)
        os.mkdir(new_dir)

        p = os.path.join(original_training_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(p):
            try:
                filename = os.path.join(p, img)
                new_filename = os.path.join(training_dir, category, img)
                img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                written = cv2.imwrite(new_filename, new_array)
                print(f'resizing {new_filename}: {written}')
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass

    for features, label in training_data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, img_size, img_size)
    print(x.shape)
    x = x/255.0
    x = x.reshape(-1, 150, 150, 1)
    y = to_categorical(y, num_classes = 4)


# def build_model():
    ### (1) try increasing image size even more (256 is too high)
    ### set this to 128 to make it go faster:
    epochs = 50
    batch_size = 40

    model = models.Sequential()
    #
    model.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (img_size,img_size,1)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    #
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.25))
    #
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.3))
    #
    model.add(layers.Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', 
                     activation ='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.3))
    
    #
    model.add(layers.Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', 
                     activation ='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.3))
    
    # 
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation = "relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation = "softmax"))
    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


    # if not path.exists(data_dir):
    #     os.mkdir(data_dir)
    #
    # if path.exists(training_dir):
    #     shutil.rmtree(training_dir)
    #
    # os.mkdir(training_dir)

    ### (2) try changing the validation split
    ### (2) play with options of ImageDataGenerator
    ### (1) add augmentation featuers of train_datagen explicitly

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        zoom_range=0.0,
        # shear_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,  


        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0.0,
        vertical_flip=False
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    train_datagen.fit(x_train)

    history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(x_val, y_val),
                                  steps_per_epoch=x_train.shape[0] // batch_size)

    ### (3) cross validation?
    # train_generator = train_datagen.flow_from_directory(
    #     training_dir,
    #     target_size=(img_size, img_size),
    #     color_mode='grayscale',
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     #save_to_dir=training_dir,
    #     #save_prefix='',
    #     #save_format='jpeg',
    #     subset='training'
    # )
    #
    # validation_generator = train_datagen.flow_from_directory(
    #     training_dir,
    #     target_size=(img_size, img_size),
    #     color_mode='grayscale',
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     subset='validation'
    # )

    #history = model.fit(
        #x=train_generator,
        #epochs=epochs,
        #verbose=2,
        #validation_data=validation_generator,
    #)

    # history = model.fit_generator(
    #     train_generator,
    #     epochs=epochs,
    #     steps_per_epoch=57,
    #     validation_data=validation_generator,
    #     validation_steps=14)

    model.save('brain-tumor-classification.h5')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_training_data()
    # build_model()

