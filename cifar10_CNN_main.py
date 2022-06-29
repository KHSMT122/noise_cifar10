# -*- coding: utf-8 -*-

import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.callbacks import CSVLogger
import cv2 as cv
from keras.models import Model
import os
from Noise import CustomImageDataGenerator
import numpy as np

# tensorboard setting
# reference for https://www.tensorflow.org/tensorboard/get_started?hl=ja
import tensorflow as tf
import datetime

class Main():
    def __init__(self, log_dir, num_classes, batch_size, ratio, epoch, validation_split, spratio=0.5, filter="", log=""):
        self.log_dir = log_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.ratio = float(ratio)
        self.epoch = epoch
        self.validation_split = validation_split
        self.filter = filter
        assert 0.0 < ratio <= 1.0, "choose ratio from 0 to 1"
        if self.filter == "none":
            self.log_dir = self.log_dir + "/" + str(self.validation_split) + "_" + self.filter
        else:
            self.log_dir = self.log_dir + "/" + str(self.validation_split) + "_" + self.filter + "_" + str(self.ratio)
        self.log = log
        if self.filter == "saltpepper":
            self.addsaltpepper = CustomImageDataGenerator()
            self.spratio = spratio

    def load(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        self.y_train = y_train
        self.y_test = y_test
        return X_train, X_test
        
    def preprocess(self):
        # normalization
        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255
        # convert onehot
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def select_filter(self, X_train, X_test):
        num_traindata = int(len(X_train) - len(X_train) * self.validation_split)
        num_filtered_image = int(self.ratio * num_traindata)
        self.X_train = []
        self.X_test = []
        print("*"*80)
        if self.filter == "none":
            print("no filter")
        elif self.filter == "median":
            self.median(num_filtered_image, num_traindata, X_train, X_test)
        elif self.filter == "average":
            self.average(num_filtered_image, num_traindata, X_train, X_test)
        elif self.filter == "gaussian":
            self.gauss(num_filtered_image, num_traindata, X_train, X_test)
        elif self.filter == "bilateral":
            self.bilateral(num_filtered_image, num_traindata, X_train, X_test)
        elif self.filter == "saltpepper":
            self.saltpepper(num_filtered_image, num_traindata, X_train, X_test)
        else:
            print("Invalid keys. Check filter name")
            exit()
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        path = "./test.png"
        cv.imwrite(path, self.X_test[0])
        print("save the converted image in", path)
        print("*"*80)

    def median(self, num_filtered_image, num_traindata, X_train, X_test):
        print("median filter")
        # テストデータには全てフィルタをかける
        for i in X_test: 
            i = cv.medianBlur(i, 3)
            self.X_test.append(i)
        # trainデータのtrainにはフィルタをかける分だけかける
        for i in X_train[:num_filtered_image]:
            i = cv.medianBlur(i, 3)
            self.X_train.append(i)
        # trainデータのvalidには全てフィルタをかける
        for i in X_train[num_traindata:]:
            i = cv.medianBlur(i, 3)
            self.X_train.append(i)
    
    def average(self, num_filtered_image, num_traindata, X_train, X_test):
        print("average filter")
        for i in X_test:
            i = cv.blur(i, (3, 3))
            self.X_test.append(i)
        for i in X_train[:num_filtered_image]:
            i = cv.blur(i, (3, 3))
            self.X_train.append(i)
        for i in X_train[num_traindata:]:
            i = cv.blur(i, (3, 3))
            self.X_train.append(i)
    
    def gauss(self, num_filtered_image, num_traindata, X_train, X_test):
        print("gaussian filter")
        for i in X_test:
            i = cv.GaussianBlur(i, (3, 3), 0)
            self.X_test.append(i)
        for i in X_train[:num_filtered_image]:
            i = cv.GaussianBlur(i, (3, 3), 0)
            self.X_train.append(i)
        for i in X_train[num_traindata:]:
            i = cv.GaussianBlur(i, (3, 3), 0)
            self.X_train.append(i)
    
    def bilateral(self, num_filtered_image, num_traindata, X_train, X_test):
        print("bilateral filter")
        for i in X_test:
            i = cv.bilateralFilter(i, 3, 100, 100)
            self.X_test.append(i)
        for i in X_train[:num_filtered_image]:
            i = cv.bilateralFilter(i, 3, 100, 100)
            self.X_train.append(i)
        for i in X_train[num_traindata:]:
            i = cv.bilateralFilter(i, 3, 100, 100)
            self.X_train.append(i)

    def saltpepper(self, num_filtered_image, num_traindata, X_train, X_test):
        print("saltpepper noise")
        for i in X_test:
            i = self.addsaltpepper.addSaltPepperNoise(src=i, amount=self.spratio)
            self.X_test.append(i)
        for i in X_train[:num_filtered_image]:
            i = self.addsaltpepper.addSaltPepperNoise(src=i, amount=self.spratio)
            self.X_train.append(i)
        for i in X_train[num_traindata:]:
            i = self.addsaltpepper.addSaltPepperNoise(src=i, amount=self.spratio)
            self.X_train.append(i)

    def backbone(self):
        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
        x = Dropout(0.5, name='block1_dropout')(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = Dropout(0.5, name='block2_dropout')(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = Dropout(0.5, name='block3_dropout')(x)

        flattened = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', name='fc1')(flattened)
        x = Dropout(0.5, name='dropout')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=inputs, outputs=predictions)

        return model
    
    def train(self):
        # print config
        print("*"*80)
        print("Batch size is", self.batch_size)
        print("Number of epoch is", self.epoch)
        print("Filter name is", self.filter)
        if self.filter == "none":
            pass
        else:
            print("Ratio of the filtered image is", self.ratio)
        print("Ratio of train and valid is", self.validation_split)
        print("*"*80)

        # load input
        X_train, X_test = self.load()

        # choose filter and ratio
        self.select_filter(X_train, X_test)

        # select format csv or tensorboard to save
        if self.log == "csv":
            filepath = self.log_dir + '/log.csv'
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
                with open(filepath, mode='w') as f:
                    f.write('')
            csv_logger = CSVLogger(filepath, separator=',', append=False)
        elif self.log == "tensorboard":
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        # preprocess
        self.preprocess()

        # define model
        model = self.backbone()

        # compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # train model
        model.fit(self.X_train, 
                  self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epoch,
                  verbose=1,
                  validation_split=self.validation_split,
                  callbacks=[csv_logger])
        
        # evaluate model
        score = model.evaluate(self.X_test, self.y_test, verbose=1)
        print('test_accuracy=', score[1], 'test_loss=', score[0])

if __name__=="__main__":
    a = Main(log_dir = "logs/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
             num_classes = 10,        # num of classes
             batch_size = 64,         # batch size
             ratio = 0.2,             # ratio of filtered traindata(not including validationdata) (float)
             epoch = 100,             # num of epoch
             validation_split = 0.1,  # splitation ratio of train and valid data (float)
             spratio = 0.5,           # necessary to use saltpepper noise (float)
             filter = "average",         # filter type ("median", "average", "gaussian", "bilateral", "saltpepper")
             log = "csv")             # log type (csv or tensorboard)
    a.train()
