import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib as mpl

########## Required for Clean Printing ###########
from pprint import pprint

########## Required for Parallel Processing ###########

from MCAIncludes import *
from joblib import Parallel, delayed
import multiprocessing
import time

import copy


########## Require for Tree Expansion ###################

def GenerateImageVariations_Maxima(image):
    ######### Global Variables ###########################

    NEIGHBORS = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, +1]])

    # Test Image

    # image = misc.face(gray=True)
    # image = misc.imresize(image,(64,64))
    TotalParallelExecutionTime = 0

    # image= np.array([[20,27,24,26],[16,23,30,32],[22,22,20,19],[22,10,35,19]])
    # print(type(image))
    # print(image.shape)

    ############ Calculating LBP of the Image ####################
    start_time = time.time()
    # -- Takes a numpy array and append rows and columns, i.e new image is of dimensions [m+2,n+2]
    iData = LoadImage(image)
    row = iData[0]
    col = iData[1]
    cImage = iData[2]
    print(cImage.shape)
    # -- calculates LBP for each of the pixel in the image
    # -- Following data structure
    # -- [i,j,pixelval,lbpval,[constraintval]] where constraint value are can be 0 (equal), 1 (greater), and -1 (smaller)
    PixelLBP = [calculateLBP(i, j, NEIGHBORS, cImage) for i, j in np.ndindex(cImage.shape) if
                i > 0 and i < row - 1 and j > 0 and j < col - 1]

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    pprint(PixelLBP)

    ####################### Boiler Plate for Plateau, Minima, and Maxima Extraction ###############
    start_time = time.time()
    # -- create a data structure to maintain the plateaus in the image
    # -- key : (i,j) - pixel index
    # -- value : [(i1,j1),..,(in,jn)] - List of co-ordinates in the image
    platoDict = getConstNeigh(PixelLBP, NEIGHBORS, row, col, 0)
    MinimaRefereces = getConstNeigh(PixelLBP, NEIGHBORS, row, col, 1)
    MaximaRefereces = getConstNeigh(PixelLBP, NEIGHBORS, row, col, -1)
    pprint(platoDict)
    pprint(MinimaRefereces)
    pprint(MaximaRefereces)

    conMatplateaus = {}
    for pixel in PixelLBP:
        conMatplateaus[(pixel[0] - 1, pixel[1] - 1)] = pixel[4]
    pprint(conMatplateaus)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    ####################### Code for Plateau, Minima, and Maxima Extraction ###############
    start_time = time.time()
    pData = ExtractPlateau(image, platoDict, conMatplateaus)
    plateuPixel = pData[0]
    plateutree = pData[1]
    PointIndex = pData[2]
    minima = pData[3]
    maxima = pData[4]
    plateutreeVal = pData[5]

    pprint(plateuPixel)
    pprint(plateutree)
    pprint(PointIndex)
    pprint(minima)
    pprint(maxima)
    pprint(plateutreeVal)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    ####################### Code for Tree Expansion - Minima ###############

    num_cores = multiprocessing.cpu_count()

    # print(num_cores, " cores are availble for execution")

    start_time = time.time()
    maxForest_Parallel = Parallel(n_jobs=-1, backend="multiprocessing", batch_size=num_cores, pre_dispatch='1.5*n_jobs',
                                  verbose=5) \
        (delayed(expandTree_Maxima)(maxima[i], MaximaRefereces, plateutree, PointIndex) for i in range(0, len(maxima)))

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    #################### Code for Image Generation _L1 #################
    start_time = time.time()

    PassImages_L1_Maxima = CreateImageFromTree_Maxima(maxForest_Parallel, row, col, plateutree)
    pass1_Image_Maxima = PassImages_L1_Maxima[0]
    pass1_LevelImage_Maxima = PassImages_L1_Maxima[1]

    pprint(pass1_Image_Maxima)
    pprint(pass1_LevelImage_Maxima)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    ###########  Add Code for Updating the root of the tree #############
    ## After max assignmet
    ## for each tree determine the max value and tree number
    ## take difference of max value with depth and update tree root.
    ## Fill the tree with the new root value and update the tree.
    maxForest_L2_Parallel = []
    maxForest_L3_Parallel = []
    maxForest_L4_Parallel = []
    for index, maxTree in enumerate(maxForest_Parallel):
        maxDepth = []
        rootNodeID = -1
        for key in maxTree.keys():
            maxDepth.append(maxTree[key][0][1])
            if maxTree[key][0][0] == 0:
                rootNodeID = key

        # for rootNodeID go into the original image and fetch the actual value of the local minima
        # and set it to maxTree[rootNodeID][0][1] and pass it to update tree depth to generate the image, the one
        # with root node initialzed to actual value of the minima.
        pprint(maxTree)
        pprint(rootNodeID)
        # We are subtracting 255 because root of the tree is initialized with 255 and at the depth of the true will be a value equal to 255 - depth of the tree
        treeDepth = 255 - min(maxDepth)
        rootValueFromTreeDepth = min(maxDepth)
        stepSize = 1
        maximaActualValue = plateutreeVal[rootNodeID]
        MaxValueforMinima  = (255/stepSize) - abs(treeDepth - maximaActualValue)

        maxTree_L2 = copy.deepcopy(maxTree)
        maxTree_L3 = copy.deepcopy(maxTree)
        maxTree_L4 = copy.deepcopy(maxTree)

        # Initializing the tree root with the Max Depth Value
        maxTree_L2[rootNodeID][0][1] = rootValueFromTreeDepth
        maxTree_L2 = UpdateTreeDepth_Minima(rootNodeID, maxTree_L2)
        maxForest_L2_Parallel.append(maxTree_L2)

        # Initializing the tree root with the Actual Value of Minima
        maxTree_L3[rootNodeID][0][1] = maximaActualValue
        maxTree_L3 = UpdateTreeDepth_Minima(rootNodeID, maxTree_L3)
        maxForest_L3_Parallel.append(maxTree_L3)

        # Initializing the tree root with the Value derived from Depth, Actual Minima and Step Size
        maxTree_L4[rootNodeID][0][1] = MaxValueforMinima
        maxTree_L4 = UpdateTreeDepth_Minima(rootNodeID, maxTree_L4, stepSize)
        maxForest_L4_Parallel.append(maxTree_L4)

    #################### Code for Image Generation _L2 #################
    start_time = time.time()

    PassImages_L2 = CreateImageFromTree_Minima(maxForest_L2_Parallel, row, col, plateutree)
    pass2_Image_Minima = PassImages_L2[0]
    pass2_LevelImage_Minima = PassImages_L2[1]

    pprint(pass2_Image_Minima)
    pprint(pass2_LevelImage_Minima)

    PassImages_L3 = CreateImageFromTree_Minima(maxForest_L3_Parallel, row, col, plateutree)
    pass3_Image_Minima = PassImages_L3[0]
    pass3_LevelImage_Minima = PassImages_L3[1]

    pprint(pass3_Image_Minima)
    pprint(pass3_LevelImage_Minima)

    PassImages_L4 = CreateImageFromTree_Minima(maxForest_L4_Parallel, row, col, plateutree)
    pass4_Image_Minima = PassImages_L4[0]
    pass4_LevelImage_Minima = PassImages_L4[1]

    pprint(pass4_Image_Minima)
    pprint(pass4_LevelImage_Minima)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    #################### Compare if the generated image and the orignal image are same - L1 #################
    start_time = time.time()

    originalLBP = GenerateLBPImage(image, NEIGHBORS)
    pass1LBP = GenerateLBPImage(pass1_Image_Minima, NEIGHBORS)

    if ((originalLBP == pass1LBP).all()):
        print("Success L1 generated")
    else:
        print("Failure L1 generated")

    TotalParallelExecutionTime = TotalParallelExecutionTime + (time.time() - start_time)

    #################### Compare if the generated image and the orignal image are same - L2 #################
    start_time = time.time()

    originalLBP = GenerateLBPImage(image, NEIGHBORS)
    pass2LBP = GenerateLBPImage(pass2_Image_Minima, NEIGHBORS)

    if ((originalLBP == pass2LBP).all()):
        print("Success L2 generated")
    else:
        print("Failure L2 generated")


    return [pass1_Image_Minima, pass2_Image_Minima, pass3_Image_Minima, pass4_Image_Minima]



image = np.array([[20, 27, 24, 26], [16, 23, 30, 32], [22, 22, 20, 19], [22, 10, 35, 19]])
print(type(image))
print(image.shape)
print(image)

lena = scipy.misc.imread('lena.png', mode='F')
image = scipy.misc.imresize(lena, (64, 64))

ImgLst = GenerateImageVariations_Minima(image)

pass1_Image_Minima=ImgLst[0]
pass2_Image_Minima=ImgLst[1]
pass3_Image_Minima=ImgLst[2]
pass4_Image_Minima=ImgLst[3]

imageLabel = 1

print("Image Label is : " + str(imageLabel))
plt.figure(1)
plt.title("Image Label is : " + str(imageLabel))

ax = plt.subplot(231)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image, cmap=mpl.cm.gray)

ax = plt.subplot(232)
ax.set_title("Minima Initialized with 0", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(233)
ax.set_title("Minima Initialized with Tree Depth", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(234)
ax.set_title("Minima Initialized with Actual Value", fontsize=7)
ax.set_axis_off()
ax.imshow(pass3_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(235)
ax.set_title("Minima Initialized with Upper Bound", fontsize=7)
ax.set_axis_off()
ax.imshow(pass4_Image_Minima, cmap=mpl.cm.gray)


plt.show()