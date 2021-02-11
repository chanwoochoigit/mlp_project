import sys

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling2D, Conv2D, Flatten, Dense, Conv3D
from tensorflow.python.keras.models import save_model, Sequential
from tensorflow.keras.optimizers import Adam
import imblearn
from preprocess import get_idx_and_find_data
from preprocess import unpickle

def fetch_and_save_data():
    labels = unpickle('cifar_100/meta')[b'fine_label_names']
    non_chair_data = []
    chair_data = []

    counter = 0
    for lbl in labels:
        print("fetching {} data...{}/{}".format(lbl, counter, len(labels)))
        if lbl != b'chair':
            dt = get_idx_and_find_data(lbl)
            non_chair_data.append(np.array(dt))
        else:
            chair_data = get_idx_and_find_data(lbl)
        counter += 1

    np.save('classifier_data/non_chair_data.npy', np.array(non_chair_data))
    np.save('classifier_data/chair_data.npy', np.array(chair_data))

def check_bias(y):
    count_0 = 0
    count_1 = 0
    for num in y:
        if num == 0:
            count_0 += 1
        elif num == 1:
            count_1 += 1
        else:
            sys.exit("wrong label input!")

    return count_0 == count_1

if __name__ == '__main__':

    # set_data()
    chair_data = np.load('classifier_data/chair_data.npy')
    non_chair_data = np.load('classifier_data/non_chair_data.npy')
    non_chair_data = non_chair_data.reshape(49500, 3072)
    cnc_data = np.vstack((non_chair_data, chair_data))
    labels = np.array([0] * 49500 + [1] * 500)

    x = cnc_data
    y = labels
    oversample = SMOTE()
    x, y = oversample.fit_resample(x, y)
    print(check_bias(y))
    x = x.reshape(99000, 32, 32, 3)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = Sequential()
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color

    kernel_size = (3,3)
    pool_size = (2,2)

    # This is the first convolution
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization(center=True, scale=True))

    # The second convolution
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization(center=True, scale=True))

    # The third convolution
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization(center=True, scale=True))

    # The fourth convolution
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization(center=True, scale=True))

    # The fifth convolution
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization(center=True, scale=True))

    # Flatten the results to feed into a DNN
    model.add(Flatten())

    # 512 neuron hidden layer
    model.add(Dense(32, activation='relu'))

    # Only 1 output neuron. 0 for non_chair 1 for chair
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics='accuracy')

    history = model.fit(x_train, y_train, epochs=15, batch_size=12, verbose=1, validation_data=(x_test,y_test))
    print(history.history)

    model.evaluate(x_test, y_test)

    save_model(model, 'models/chair_classifier')