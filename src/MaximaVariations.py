########## Required for Load Image and Display ###########

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib as mpl


########## Required for Generating Variations ###########

from MCAIncludes import *


# Test Image
#image = np.array([[20, 27, 24, 26], [16, 23, 30, 32], [22, 22, 20, 19], [22, 10, 35, 19]])
#print(image.shape)

# Load Image from Current Directory
lena = scipy.misc.imread('../assets/lena.png', mode='F')

# Resize
image = scipy.misc.imresize(lena, (64, 64))

# Call Genrate Variation from MCAIncludes
ImgLst = GenerateImageVariations_Maxima(image)
pass1_Image_Maxima = ImgLst[0]
pass2_Image_Maxima = ImgLst[1]
pass3_Image_Maxima = ImgLst[2]
pass4_Image_Maxima = ImgLst[3]

# Create a visualization
imageLabel = "Lena - The CV Girl"

print("Image Label is : " + str(imageLabel))
plt.figure(1)
plt.title("Image Label is : " + str(imageLabel))

ax = plt.subplot(231)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image, cmap=mpl.cm.gray)

ax = plt.subplot(232)
ax.set_title("Maxima Initialized with 0", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(233)
ax.set_title("Maxima Initialized with Tree Depth", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(234)
ax.set_title("Maxima Initialized with Actual Value", fontsize=7)
ax.set_axis_off()
ax.imshow(pass3_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(235)
ax.set_title("Maxima Initialized with Upper Bound", fontsize=7)
ax.set_axis_off()
ax.imshow(pass4_Image_Maxima, cmap=mpl.cm.gray)

plt.show()
