# Refering http://deepdish.io/2014/11/11/python-dictionary-to-hdf5/ to read and write files
from keras.datasets import cifar10
import numpy as np
import deepdish as dd
from MCAIncludes import *
import os


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    grayImage = np.around(gray)

    return grayImage


def imgGenerator(grayImage):
    ImgLst = GenerateImageVariations_Minima(grayImage)
    pass1_Image_Minima = ImgLst[0]
    pass2_Image_Minima = ImgLst[1]
    pass3_Image_Minima = ImgLst[2]
    pass4_Image_Minima = ImgLst[3]
    min_avg = ((pass1_Image_Minima + pass2_Image_Minima + pass3_Image_Minima + pass4_Image_Minima) / 4)

    ImgLst = GenerateImageVariations_Maxima(grayImage)
    pass1_Image_Maxima = ImgLst[0]
    pass2_Image_Maxima = ImgLst[1]
    pass3_Image_Maxima = ImgLst[2]
    pass4_Image_Maxima = ImgLst[3]
    max_avg = ((pass1_Image_Maxima + pass2_Image_Maxima + pass3_Image_Maxima + pass4_Image_Maxima) / 4)

    pass1_image_avg = ((pass1_Image_Minima + pass1_Image_Maxima) / 2)
    pass2_image_avg = ((pass2_Image_Minima + pass2_Image_Maxima) / 2)
    pass3_image_avg = ((pass3_Image_Minima + pass3_Image_Maxima) / 2)
    pass4_image_avg = ((pass4_Image_Minima + pass4_Image_Maxima) / 2)
    min_max_avg = ((min_avg + max_avg) / 2)

    return [pass1_Image_Minima, pass2_Image_Minima, pass3_Image_Minima, pass4_Image_Minima, min_avg, \
            pass1_Image_Maxima, pass2_Image_Maxima, pass3_Image_Maxima, pass4_Image_Maxima, max_avg, \
            pass1_image_avg, pass2_image_avg, pass3_image_avg, pass4_image_avg, min_max_avg]


def GenerateAllData(numRandomSamplesTrain=1,numRandomSamplesTest=1):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    data = x_train
    labels = y_train

    print(data.shape)
    print(labels.shape)

    print(x_test.shape)
    print(y_test.shape)


    # Randomly selecting numRandomSamplesTrain=1 samples from each class and generating 15 new contrast varying sample

    for i in range(0, 10):

        # initialize image set for all the 15 variations
        _trainSet = {}
        for h in range(0, 16):
            _trainSet[h] = []

        # filter labels of a given class , and draw a random sample of n images , images are stored in temp_data while
        # labels on temp_labels
        filtered_labels = np.where(labels == i)[0]
        temp_labels = np.random.choice(filtered_labels, numRandomSamplesTrain)
        temp_data = data[temp_labels]

        # for each of the image in temp_data, convert it into grayscale and generate contrast variations
        # for each of the variation append contrast variation into respective trainset {key:0 value: [list of images of
        # variations with type 0]}
        for j in range(0, len(temp_labels)):
            grayImage = rgb2gray(temp_data[j])
            imgLst = imgGenerator(grayImage)
            _trainSet[0].append(grayImage)
            for k in range(0, 15):
                _trainSet[k + 1].append(imgLst[k])

        # Now we have all the variation generated for a specific image class, we create a datastructure to store the
        # training images and their labels
        for p in range(0, 16):
            _dataDict = {}

            x_data = np.array(_trainSet[p])
            print(x_data.shape)

            _dataDict['Data'] = x_data
            _dataDict['Labels'] = labels[temp_labels]

            # Sample ./DATA/0/data_batch_0_3.h5
            filedir = './DATA/' + str(i)
            filename = filedir + '/data_batch_' + str(i) + '_' + str(p) + '.h5'
            try:
                os.makedirs(filedir)
            except FileExistsError:
                # directory already exists
                pass
            dd.io.save(filename, _dataDict, compression=None)


    testData = []
    testLabels = []

    for i in range(0, 10):
        filtered_labels = np.where(y_test == i)[0]
        temp_labels = np.random.choice(filtered_labels, numRandomSamplesTest)
        temp_data = x_test[temp_labels]

        for j in range(0, len(temp_labels)):
            grayImage = rgb2gray(temp_data[j])
            testData.append(grayImage)
            testLabels.append(y_test[temp_labels[j]])

    _dataDict = {}
    _dataDict['Data'] = np.array(testData)
    _dataDict['Labels'] = np.array(testLabels)

    print(_dataDict['Data'].shape)
    print(_dataDict['Labels'].shape)

    filedir = './DATA/test'
    filename = filedir + '/data_batch_test.h5'
    try:
        os.makedirs(filedir)
    except FileExistsError:
        # directory already exists
        pass
    dd.io.save(filename, _dataDict, compression=None)



GenerateAllData(1,1)
# GenerateAllData(5000,1000) -- To run at office

