from keras.datasets import cifar10
import numpy as np


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.around(gray)

    return grayImage


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# need to randomly select 670 samples from each class for each of them need to generate 15 samples from contrast
# variations and use it as an input to the pipeline.

for i in range(0, 50):
    print(y_train[i][0])
    print(x_train[i].shape)
    img = rgb2gray(x_train[i])
    print(img.shape)
