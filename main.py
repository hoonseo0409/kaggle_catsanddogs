import os, cv2, random
import numpy as np
import pandas as pd
import time
import csv

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras import regularizers

import tensorflow as tf

with tf.device('/cpu:0'):
    # K.set_image_dim_ordering('th')

    TRAIN_CATS_DIR = 'input/train/cats/'
    TRAIN_DOGS_DIR = 'input/train/dogs/'
    TEST_DIR = 'input/test/'
    VALD_CATS_DIR = 'input/validation/cats/'
    VALD_DOGS_DIR = 'input/validation/dogs/'
    AUG_DIR = 'input/train/aug/'

    ROWS = 128
    COLS = 128
    CHANNELS = 3

    train_dogs = [TRAIN_DOGS_DIR + i for i in os.listdir(TRAIN_DOGS_DIR)]
    train_cats = [TRAIN_CATS_DIR + i for i in os.listdir(TRAIN_CATS_DIR)]
    vald_dogs = [VALD_DOGS_DIR + i for i in os.listdir(VALD_DOGS_DIR)]
    vald_cats = [VALD_CATS_DIR + i for i in os.listdir(VALD_CATS_DIR)]

    # 주의 : 개 고양이 섞고 트레인으로 합치는거였는데, 개 고양이 놔두고 합친 뒤 트레인을 섞는 걸로 바꿈 : 오그멘테이션 때문에
    # random.shuffle(train_cats)
    # random.shuffle(train_dogs)

    train_images = train_cats + train_dogs  # use this for full dataset
    random.shuffle(train_images)
    vald_images = vald_cats + vald_dogs
    random.shuffle(vald_images)
    print(train_images)

    test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
    # random.shuffle(test_images)
    print(test_images)


    def read_image(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
        return img


    def prep_data(images):
        count = len(images)
        # data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
        data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
        for i, image_file in enumerate(images):
            image = read_image(image_file)
            # data[i] = (np.array (image, dtype=np.float32)).T
            # data[i] = (np.array(image, dtype=np.uint8)).T
            # data[i] = (np.array(image, dtype=np.uint8).astype(float)).T
            data[i] = (img_to_array(image)).T
            if i % 250 == 0: print('Processed {} of {}'.format(i, count))
        return data


    train = prep_data(train_images)
    test = prep_data(test_images)
    vald = prep_data(vald_images)

    sizeOfaug = round(len(train) / 50)

    # 이미지 오그멘테이션 생성

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest', featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True,
        vertical_flip=True)

    # 오그멘테이션 이미지 생성
    print("sizeOfaug 값은:", sizeOfaug)
    for i in train_images[:sizeOfaug]:
        # print("디렉토리 잘 나오나: ", i)
        img = read_image(i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        # fit 삭제함 멈춰서 여기에 fit 넣어야 zca화이트닝 나오니까
        if 'dogs' in i:
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='input/train/aug', save_prefix='dogs', save_format='jpg'):
                break  # otherwise the generator would loop indefinitely
        elif 'cats' in i:
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='input/train/aug', save_prefix='cats', save_format='jpg'):
                break  # otherwise the generator would loop indefinitely

    # 오그멘테이션 이미지 학습 세트에 추가
    aug_images = [AUG_DIR + i for i in os.listdir(AUG_DIR)]
    random.shuffle(aug_images)
    aug = prep_data(aug_images)
    print("train의 모양이 이랬는데: ", train.shape)
    train = np.append(train, aug, axis=0)
    print("train의 모양이 이렇게 바뀜: ", train.shape)
    # print("트레인은: :", train)


    print("Train shape: {}".format(train.shape))
    print("Test shape: {}".format(test.shape))

    # 이미지 라벨링
    labels = []
    for i in train_images:
        if 'dogs' in i:
            labels.append(1)
        elif 'cats' in i:
            labels.append(0)
        else:
            print("에러!!!: cats도 아니고 dogs도 아닌 그림 파일이 검출됨")

    for i in aug_images:
        if 'dogs' in i:
            labels.append(1)
        elif 'cats' in i:
            labels.append(0)
        else:
            print("aug 에러!!!: cats도 아니고 dogs도 아닌 그림 파일이 검출됨")

    vald_labels = []
    for i in vald_images:
        if 'dogs' in i:
            vald_labels.append(1)
        elif 'cats' in i:
            vald_labels.append(0)
        else:
            print("vald 에러!!!: cats도 아니고 dogs도 아닌 그림 파일이 검출됨")

    """sns.countplot(labels)
    plt.title('Cats and Dogs')
    plt.show()"""

    start = time.time()
    optimizer = RMSprop(lr=1e-4)
    objective = 'binary_crossentropy'

with tf.device('/gpu:0'):
    def catdog():
        # 학습 관련 파라메타들

        sizeOffm = 4
        sizeOfpm = 3
        my_maxnorm = 2.

        model = Sequential()

        # model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(3, ROWS, COLS), kernel_constraint=maxnorm(2),  activation='relu'))
        model.add(Convolution2D(32, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm), activation='relu'))
        model.add(Convolution2D(32, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(64, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(Convolution2D(64, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(128, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(Convolution2D(128, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(256, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        # model.add(Convolution2D(256, 3, 3, border_mode='same', kernel_constraint=maxnorm(1), activation='relu'))
        model.add(Convolution2D(256, sizeOffm, sizeOffm, border_mode='same', kernel_constraint=maxnorm(my_maxnorm),
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, kernel_constraint=maxnorm(my_maxnorm), activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, kernel_constraint=maxnorm(my_maxnorm), activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
        model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
        return model


    model = catdog()

    nb_epoch = 20
    batch_size = 128


    ## Callback for loss logging per epoch
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))


    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')


    def run_catdog():
        history = LossHistory()
        print(labels)
        model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_split=0.25, verbose=2, shuffle=True, callbacks=[history, early_stopping])

        predictions = model.predict(test, verbose=0)
        return predictions, history


    predictions, history = run_catdog()

    loss = history.losses
    val_loss = history.val_losses

with tf.device('/cpu:0'):
    model.save_weights('first_try.h5')

    print("연산시간은 : ", time.time() - start)

    # accuracy 측정
    scores = model.evaluate(vald, vald_labels, verbose=0)
    print("Baseline Accuracy: %.2f%%" % (scores[1] * 100))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VGG-16 Loss Trend')
    plt.plot(loss, 'blue', label='Training Loss')
    plt.plot(val_loss, 'green', label='Validation Loss')
    plt.xticks(range(0, nb_epoch)[0::2])
    plt.legend()
    plt.show()

    """for i in range(0, 10):
        if predictions[i, 0] >= 0.5:
            print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
        else:
            print('I am {:.2%} sure this is a Cat'.format(1 - predictions[i][0]))

        plt.imshow(test[i].T)
        plt.show()"""

    with open('./result.csv', 'w', newline="\n") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id'] + ['Category'])
        for i in range(0, 1962):
            if predictions[i][0] >= 0.5:
                pred = 1
            else:
                pred = 0
            filename = test_images[i][14:]
            writer.writerow([filename] + [pred])