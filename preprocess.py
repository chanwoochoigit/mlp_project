import sys

import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_index(word):
    word_idx = -1
    labels = unpickle('cifar_100/meta')[b'fine_label_names']
    for i in range(len(labels)):
        if labels[i] == word:
            word_idx = i

    if word_idx != -1:
        return word_idx
    else:
        sys.exit("given word not found!")

def get_data(word_idx):

    big_data = unpickle('cifar_100/train')
    data_labels = big_data[b'fine_labels']
    data = big_data[b'data']

    indices = []
    for i in range(len(data_labels)):
        print("finding labels...{}/{}".format(i,len(data_labels)))
        if data_labels[i] == word_idx:
            indices.append(i)

    corresponding_data = []
    for idx in indices:
        corresponding_data.append(np.array(data[idx]))

    if len(indices) != len(corresponding_data):
        sys.exit("Potential corrosion in the indices list!")
    else:
        return np.array(corresponding_data)

def generate_and_save_images(pixel_data, label):

    for i in range(len(pixel_data)):
        red_layer = pixel_data[i][:1024]
        green_layer = pixel_data[i][1024:2048]
        blue_layer = pixel_data[i][2048:]

        rgb_array = []

        for rgb_layer in zip(red_layer,green_layer,blue_layer):
            rgb_array.append(np.array(rgb_layer))

        rgb_array = np.array(rgb_array).reshape(32,32,3)

        image = Image.fromarray(rgb_array)
        image.save('images/{}_{}.png'.format(label,i))
    print("{} images saved!".format(label))

def get_idx_and_find_data(word):
    word_idx = get_index(word)
    data = get_data(word_idx)
    return data

if __name__ == '__main__':

    chair_data = get_idx_and_find_data(b'chair')
    bee_data = get_idx_and_find_data(b'bee')
    cloud_data = get_idx_and_find_data(b'cloud')
    mushroom_data = get_idx_and_find_data(b'mushroom')

    generate_and_save_images(chair_data, 'chair')
    generate_and_save_images(bee_data, 'bee')
    generate_and_save_images(cloud_data, 'cloud')
    generate_and_save_images(mushroom_data, 'mushroom')