# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import glob
# from keras import layers
# from keras import models
from os import path
import os
import shutil

base_dir = "/home/peter/dev/brain-tumors"
data_dir = base_dir + "/data"  # use os.path.join
training_dir = data_dir + "/Training"
validation_dir = data_dir + "/Validation"
categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def prepare_data():

    if not path.exists(data_dir):
        print(f'{data_dir} does not exist, preparing data...', end=' ')
        original_data_dir = base_dir + "/original-data"
        shutil.copytree(original_data_dir, data_dir)
        os.mkdir(validation_dir)

        # Move 100 files of each dir to validation data
        for category in categories:
            current_dir = training_dir + "/" + category
            validation_dir_cat = validation_dir + "/" + category
            os.mkdir(validation_dir_cat)

            for f in glob.glob(current_dir + "/*.jpg")[:100]:
                shutil.move(f, validation_dir_cat)

        print('done')


def load_images():
    testing_dir = data_dir + "/Testing"
    print(f'here load images')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepare_data()
    load_images()

