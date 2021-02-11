import sys

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v1 import adam
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

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # model = tf.keras.models.Sequential([
    # # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    #
    # # This is the first convolution
    # tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    #
    # # The second convolution
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    #
    # # The third convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu')
    # tf.keras.layers.MaxPooling2D(2,2),
    #
    # # The fourth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    #
    # # The fifth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    #
    # # Flatten the results to feed into a DNN
    # tf.keras.layers.Flatten(),
    #
    # # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    #
    # # Only 1 output neuron. 0 for non_chair 1 for chair
    # tf.keras.layers.Dense(1, activation='sigmoid')])
    #
    # model.compile(loss='binary_crossentropy', optimizer=adam(lr=0.001), metrics='accuracy')
    #
    # train_test_split
    # history = model.fit(x_train, steps_per_epoch=8, epochs=15, verbose=1, validation_data=x_test,
    #                     validation_steps=8)