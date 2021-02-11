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
        if i % 10000 == 0:
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

        rgb_array = np.array(rgb_array).reshape((32,32,3))

        image = Image.fromarray(rgb_array)
        image.save('images/{}_{}.png'.format(label,i))
    print("{} images saved!".format(label))

def get_idx_and_find_data(word):
    word_idx = get_index(word)
    data = get_data(word_idx)
    return data

# def concat_helper(data_1, data_2, data_3):
#     new_data = []
#
#     print("_______________________________________")
#     for i in range(len(data_1)):
#         print("mixing data...{}/{}".format(i, len(data_1)))
#         for j in range(len(data_2)):
#             for k in range(len(data_3)):
#                 red = data_1[i][:1024] + data_2[j][:1024] + data_3[k][:1024]
#                 green = data_1[i][1024:2048] + data_2[j][1024:2048] + data_3[k][1024:2048]
#                 blue = data_1[i][2048:] + data_2[j][2048:] + data_3[k][2048:]
#                 new_data.append(np.array([red, green, blue]))
#     print("_______________________________________")
#     return np.array(new_data)
#
# def concat_pixel_data(data_1, data_2, data_3):
#     batch_1 = concat_helper(data_1[:5], data_2, data_3)
#     # batch_2 = concat_helper(data_1[100:200], data_2, data_3)
#     # batch_3 = concat_helper(data_1[200:300], data_2, data_3)
#     # batch_4 = concat_helper(data_1[300:400], data_2, data_3)
#     # batch_5 = concat_helper(data_1[400:], data_2, data_3)
#
#     # return batch_1 + batch_2 + batch_3 + batch_4 + batch_5
#     print("concatenation successful!")
#     return batch_1
if __name__ == '__main__':

    chair_data = get_idx_and_find_data(b'chair')
    bee_data = get_idx_and_find_data(b'bee')
    cloud_data = get_idx_and_find_data(b'cloud')
    # mushroom_data = get_idx_and_find_data(b'mushroom')

    # generate_and_save_images(chair_data, 'chair')
    # generate_and_save_images(bee_data, 'bee')
    # generate_and_save_images(cloud_data, 'cloud')
    # generate_and_save_images(mushroom_data, 'mushroom')

    bee_cloud = np.vstack((chair_data, bee_data, cloud_data))
    np.save('mixed_data/bee_cloud.npy',bee_cloud)
    # bee_cloud = np.load('mixed_data/bee_cloud.npy')
    print(bee_cloud.shape)
