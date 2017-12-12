#!/usr/bin/env python2

# coding:utf-8

from __future__ import print_function

import gc
import random
import os

import numpy as np
from keras import backend as Keras
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from utils import extract_data, resize_with_pad, IMAGE_SIZE, adfis, banner


class DataSet(object):
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None

        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, nb_classes=2):
        images, labels = extract_data('./data/')
        labels = np.reshape(labels, [-1])
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=0.3,
            random_state=random.randint(0, 100)
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            images,
            labels,
            test_size=0.5,
            random_state=random.randint(0, 100)
        )

        # Tensoflow ordering:
        assert Keras.image_dim_ordering() == 'tf'
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

        # The data, shuffled and split between train and test sets:
        adfis('X_train shape:', X_train.shape)
        adfis(X_train.shape[0], 'train samples')
        adfis(X_valid.shape[0], 'valid samples')
        adfis(X_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices:
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test

        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test


class Model(object):
    FILE_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'store/model.h5'
    )

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                32,
                (3, 3),
                padding='same',
                input_shape=dataset.X_train.shape[1:]
            )
        )
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(
            self,
            dataset,
            batch_size=32,
            nb_epoch=40,
            data_augmentation=True
    ):
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )

        if not data_augmentation:
            adfis('Not using data augmentation.')
            self.model.fit(
                dataset.X_train,
                dataset.Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                validation_data=(dataset.X_valid, dataset.Y_valid),
                shuffle=True
            )

        else:
            adfis('Using real-time data augmentation.')

            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                # Set input mean to 0 over the dataset:
                samplewise_center=False,  # Set each sample mean to 0.
                featurewise_std_normalization=False,
                # Divide inputs by std of the dataset:
                samplewise_std_normalization=False,
                # Divide each input by its std:
                zca_whitening=False,  # Apply ZCA whitening.
                rotation_range=20,
                # Randomly rotate images in the range (degrees, 0 to 180):
                width_shift_range=0.2,
                # Randomly shift images horizontally (fraction of total width):
                height_shift_range=0.2,
                # Randomly shift images vertically (fraction of total height):
                horizontal_flip=True,  # Randomly flip images.
                vertical_flip=False  # Randomly flip images.
            )

            # Compute quantities required for featurewise normalization
            # (std, mean and principal components if ZCA whitening is applied):
            datagen.fit(dataset.X_train)

            # Fit the model on the batches generated by datagen.flow():
            self.model.fit_generator(
                datagen.flow(
                    dataset.X_train,
                    dataset.Y_train,
                    batch_size=batch_size
                ),
                epochs=nb_epoch,
                steps_per_epoch=dataset.X_train.shape[0],
                validation_data=(dataset.X_valid, dataset.Y_valid)
            )

    def save(self, file_path=FILE_PATH):
        adfis('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        adfis('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        tf_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)

        assert Keras.image_dim_ordering() == 'tf' and image.shape != tf_shape
        image = resize_with_pad(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        image = image.astype('float32')
        image /= 255

        result = self.model.predict_proba(image)
        # adfis("Prediction result: {0}.".format(result))
        result = self.model.predict_classes(image)

        return result[0]

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        adfis("Evaluation: %s: %.2f%%" % (
            self.model.metrics_names[1],
            score[1] * 100)
        )


def main():
    banner()
    dataset = DataSet()
    dataset.read()

    model = Model()
    model.build_model(dataset)
    model.train(dataset, nb_epoch=10)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)
    gc.collect()


if __name__ == '__main__':
    main()
