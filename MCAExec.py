import numpy as np
import matplotlib.pyplot as plt

########## Required for Clean Printing ###########
from pprint import pprint

########## Required for Parallel Processing ###########

from MCAIncludes import *
from joblib import Parallel, delayed
import multiprocessing
import time

import copy


########## Require for Tree Expansion ###################

def GenerateImageVariations_Minima(image):
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
    # print("For Serial Execution,LBP Procedure took--- %s seconds ---" % (time.time() - start_time))
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
    minForest_Parallel = Parallel(n_jobs=-1, backend="multiprocessing", batch_size=num_cores, pre_dispatch='1.5*n_jobs',
                                  verbose=5) \
        (delayed(expandTree_Minima)(minima[i], MinimaRefereces, plateutree, PointIndex) for i in range(0, len(minima)))

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))
    pprint(minForest_Parallel)

    #################### Code for Image Generation _L1 #################
    start_time = time.time()

    PassImages_L1 = CreateImageFromTree_Minima(minForest_Parallel, row, col, plateutree)
    pass1_Image_Minima = PassImages_L1[0]
    pass1_LevelImage_Minima = PassImages_L1[1]

    pprint(pass1_Image_Minima)
    pprint(pass1_LevelImage_Minima)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    ###########  Add Code for Updating the root of the tree #############
    ## After max assignmet
    ## for each tree determine the max value and tree number
    ## take difference of max value with depth and update tree root.
    ## Fill the tree with the new root value and update the tree.
    minForest_L2_Parallel = []
    minForest_L3_Parallel = []
    minForest_L4_Parallel = []
    for index, minTree in enumerate(minForest_Parallel):
        maxDepth = []
        rootNodeID = -1
        for key in minTree.keys():
            maxDepth.append(minTree[key][0][1])
            if minTree[key][0][0] == 0:
                rootNodeID = key

        # for rootNodeID go into the original image and fetch the actual value of the local minima
        # and set it to maxTree[rootNodeID][0][1] and pass it to update tree depth to generate the image, the one
        # with root node initialzed to actual value of the minima.
        pprint(minTree)
        pprint(rootNodeID)
        stepSize = 1
        maxDepthVal = max(maxDepth)
        minimaActualValue = plateutreeVal[rootNodeID]
        MaxValueforMinima  = (255/stepSize) - abs(maxDepthVal-minimaActualValue)

        minTree_L2 = copy.deepcopy(minTree)
        minTree_L3 = copy.deepcopy(minTree)
        minTree_L4 = copy.deepcopy(minTree)

        # Initializing the tree root with the Max Depth Value
        minTree_L2[rootNodeID][0][1] = maxDepthVal
        minTree_L2 = UpdateTreeDepth_Minima(rootNodeID, minTree_L2)
        minForest_L2_Parallel.append(minTree_L2)

        # Initializing the tree root with the Actual Value of Minima
        minTree_L3[rootNodeID][0][1] = minimaActualValue
        minTree_L3 = UpdateTreeDepth_Minima(rootNodeID, minTree_L3)
        minForest_L3_Parallel.append(minTree_L3)

        # Initializing the tree root with the Value derived from Depth, Actual Minima and Step Size
        minTree_L4[rootNodeID][0][1] = MaxValueforMinima
        minTree_L4 = UpdateTreeDepth_Minima(rootNodeID, minTree_L4, stepSize)
        minForest_L4_Parallel.append(minTree_L4)

    #################### Code for Image Generation _L2 #################
    start_time = time.time()

    PassImages_L2 = CreateImageFromTree_Minima(minForest_L2_Parallel, row, col, plateutree)
    pass2_Image_Minima = PassImages_L2[0]
    pass2_LevelImage_Minima = PassImages_L2[1]

    pprint(pass2_Image_Minima)
    pprint(pass2_LevelImage_Minima)

    PassImages_L3 = CreateImageFromTree_Minima(minForest_L3_Parallel, row, col, plateutree)
    pass3_Image_Minima = PassImages_L3[0]
    pass3_LevelImage_Minima = PassImages_L3[1]

    pprint(pass3_Image_Minima)
    pprint(pass3_LevelImage_Minima)

    PassImages_L4 = CreateImageFromTree_Minima(minForest_L4_Parallel, row, col, plateutree)
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


def GenerateImageVariations(image):
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
    # print("For Serial Execution,LBP Procedure took--- %s seconds ---" % (time.time() - start_time))
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
    minForest_Parallel = Parallel(n_jobs=-1, backend="multiprocessing", batch_size=num_cores, pre_dispatch='1.5*n_jobs',
                                  verbose=5) \
        (delayed(expandTree_Minima)(minima[i], MinimaRefereces, plateutree, PointIndex) for i in range(0, len(minima)))

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))
    pprint(minForest_Parallel)

    #################### Code for Image Generation _L1 #################
    start_time = time.time()

    PassImages_L1 = CreateImageFromTree_Minima(minForest_Parallel, row, col, plateutree)
    pass1_Image_Minima = PassImages_L1[0]
    pass1_LevelImage_Minima = PassImages_L1[1]

    pprint(pass1_Image_Minima)
    pprint(pass1_LevelImage_Minima)

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    ###########  Add Code for Updating the root of the tree #############
    ## After max assignmet
    ## for each tree determine the max value and tree number
    ## take difference of max value with depth and update tree root.
    ## Fill the tree with the new root value and update the tree.
    minForest_L2_Parallel = []
    minForest_L3_Parallel = []
    minForest_L4_Parallel = []
    for index, minTree in enumerate(minForest_Parallel):
        maxDepth = []
        rootNodeID = -1
        for key in minTree.keys():
            maxDepth.append(minTree[key][0][1])
            if minTree[key][0][0] == 0:
                rootNodeID = key

        # for rootNodeID go into the original image and fetch the actual value of the local minima
        # and set it to maxTree[rootNodeID][0][1] and pass it to update tree depth to generate the image, the one
        # with root node initialzed to actual value of the minima.
        pprint(minTree)
        pprint(rootNodeID)
        stepSize = 1
        maxDepthVal = max(maxDepth)
        minimaActualValue = plateutreeVal[rootNodeID]
        MaxValueforMinima  = (255/stepSize) - abs(maxDepthVal-minimaActualValue)

        minTree_L2 = copy.deepcopy(minTree)
        minTree_L3 = copy.deepcopy(minTree)
        minTree_L4 = copy.deepcopy(minTree)

        # Initializing the tree root with the Max Depth Value
        minTree_L2[rootNodeID][0][1] = maxDepthVal
        minTree_L2 = UpdateTreeDepth_Minima(rootNodeID, minTree_L2)
        minForest_L2_Parallel.append(minTree_L2)

        # Initializing the tree root with the Actual Value of Minima
        minTree_L3[rootNodeID][0][1] = minimaActualValue
        minTree_L3 = UpdateTreeDepth_Minima(rootNodeID, minTree_L3)
        minForest_L3_Parallel.append(minTree_L3)

        # Initializing the tree root with the Value derived from Depth, Actual Minima and Step Size
        minTree_L4[rootNodeID][0][1] = MaxValueforMinima
        minTree_L4 = UpdateTreeDepth_Minima(rootNodeID, minTree_L4, stepSize)
        minForest_L4_Parallel.append(minTree_L4)

    #################### Code for Image Generation _L2 #################
    start_time = time.time()

    PassImages_L2 = CreateImageFromTree_Minima(minForest_L2_Parallel, row, col, plateutree)
    pass2_Image_Minima = PassImages_L2[0]
    pass2_LevelImage_Minima = PassImages_L2[1]

    pprint(pass2_Image_Minima)
    pprint(pass2_LevelImage_Minima)

    PassImages_L3 = CreateImageFromTree_Minima(minForest_L3_Parallel, row, col, plateutree)
    pass3_Image_Minima = PassImages_L3[0]
    pass3_LevelImage_Minima = PassImages_L3[1]

    pprint(pass3_Image_Minima)
    pprint(pass3_LevelImage_Minima)

    PassImages_L4 = CreateImageFromTree_Minima(minForest_L4_Parallel, row, col, plateutree)
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

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    ####################### Code for Tree Expansion - Maxima ###############

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

        pprint(maxTree)
        pprint(rootNodeID)
        stepSize = 1
        maxDepthVal = min(maxDepth)
        maximaActualValue = plateutreeVal[rootNodeID]
        MinValueforMaxima  = stepSize * abs(maxDepthVal-minimaActualValue)

        #maxDepthVal = min(maxDepth)
        maxTree[rootNodeID][0][1] = maxDepthVal
        maxTree_L2 = UpdateTreeDepth_Maxima(rootNodeID, maxTree)
        maxForest_L2_Parallel.append(maxTree_L2)

    #################### Code for Image Generation _L2 #################
    start_time = time.time()

    PassImages_L2_Maxima = CreateImageFromTree_Maxima(maxForest_L2_Parallel, row, col, plateutree)
    pass2_Image_Maxima = PassImages_L2_Maxima[0]
    pass2_LevelImage_Maxima = PassImages_L2_Maxima[1]

    TotalParallelExecutionTime = TotalParallelExecutionTime + ((time.time() - start_time))

    #################### Compare if the generated image and the orignal image are same - L1 #################
    start_time = time.time()

    originalLBP = GenerateLBPImage(image, NEIGHBORS)
    pass1LBP = GenerateLBPImage(pass1_Image_Maxima, NEIGHBORS)

    if ((originalLBP == pass1LBP).all()):
        print("Success L1 generated")
    else:
        print("Failure L1 generated")

    TotalParallelExecutionTime = TotalParallelExecutionTime + (time.time() - start_time)

    #################### Compare if the generated image and the orignal image are same - L2 #################
    start_time = time.time()

    originalLBP = GenerateLBPImage(image, NEIGHBORS)
    pass2LBP = GenerateLBPImage(pass2_Image_Maxima, NEIGHBORS)

    if ((originalLBP == pass2LBP).all()):
        print("Success L2 generated")
    else:
        print("Failure L2 generated")

    TotalParallelExecutionTime = TotalParallelExecutionTime + (time.time() - start_time)

    print("Total Time for Parallel Execution is : ", TotalParallelExecutionTime)

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    min_avg = ((pass1_Image_Minima + pass2_Image_Minima) / 2)
    max_avg = ((pass1_Image_Maxima + pass2_Image_Maxima) / 2)
    pass1_image_avg = ((pass1_Image_Minima + pass1_Image_Maxima) / 2)
    pass2_image_avg = ((pass2_Image_Minima + pass2_Image_Maxima) / 2)
    min_max_avg = ((min_avg + max_avg) / 2)

    return [pass1_Image_Minima, pass2_Image_Minima, min_avg, pass1_Image_Maxima, pass2_Image_Maxima, max_avg, \
            pass1_image_avg, pass2_image_avg, min_max_avg, image]


image = np.array([[20, 27, 24, 26], [16, 23, 30, 32], [22, 22, 20, 19], [22, 10, 35, 19]])
print(type(image))
print(image.shape)

print(image)

ImgLst = GenerateImageVariations(image)

pass1_Image_Minima = ImgLst[0]
pass2_Image_Minima = ImgLst[1]
min_avg = ImgLst[2]
pass1_Image_Maxima = ImgLst[3]
pass2_Image_Maxima = ImgLst[4]
max_avg = ImgLst[5]
pass1_image_avg = ImgLst[6]
pass2_image_avg = ImgLst[7]
min_max_avg = ImgLst[8]
image = ImgLst[9]
imageLabel = 1

print("Image Label is : " + str(imageLabel))
plt.figure(1)
plt.title("Image Label is : " + str(imageLabel))

ax = plt.subplot(341)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image)

ax = plt.subplot(342)
ax.set_title("Pass 1 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Minima)

ax = plt.subplot(343)
ax.set_title("Pass 2 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Minima)

ax = plt.subplot(344)
ax.set_title("Pass 1 & 2 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(min_avg)

ax = plt.subplot(345)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image)

ax = plt.subplot(346)
ax.set_title("Pass 1 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Maxima)

ax = plt.subplot(347)
ax.set_title("Pass 2 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Maxima)

ax = plt.subplot(348)
ax.set_title("Pass 1 & 2 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(max_avg)

ax = plt.subplot(349)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image)

ax = plt.subplot(3, 4, 10)
ax.set_title("Pass 1 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_image_avg)

ax = plt.subplot(3, 4, 11)
ax.set_title("Pass 2 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_image_avg)

ax = plt.subplot(3, 4, 12)
ax.set_title("Pass 1 & 2 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(min_max_avg)

plt.show()
