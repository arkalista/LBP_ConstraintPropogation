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
ImgLst = GenerateImageVariations_Minima(image)
pass1_Image_Minima = ImgLst[0]
pass2_Image_Minima = ImgLst[1]
pass3_Image_Minima = ImgLst[2]
pass4_Image_Minima = ImgLst[3]

ImgLst = GenerateImageVariations_Maxima(image)
pass1_Image_Maxima = ImgLst[0]
pass2_Image_Maxima = ImgLst[1]
pass3_Image_Maxima = ImgLst[2]
pass4_Image_Maxima = ImgLst[3]

min_avg = ((pass1_Image_Minima + pass2_Image_Minima + pass3_Image_Minima + pass4_Image_Minima) / 4)
max_avg = ((pass1_Image_Maxima + pass2_Image_Maxima + pass3_Image_Maxima + pass4_Image_Maxima) / 4)

pass1_image_avg = ((pass1_Image_Minima + pass1_Image_Maxima) / 2)
pass2_image_avg = ((pass2_Image_Minima + pass2_Image_Maxima) / 2)
pass3_image_avg = ((pass3_Image_Minima + pass3_Image_Maxima) / 2)
pass4_image_avg = ((pass4_Image_Minima + pass4_Image_Maxima) / 2)

min_max_avg = ((min_avg + max_avg) / 2)

plt.figure(1)
plt.title("Hello World - MCA")

ax = plt.subplot(361)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image, cmap=mpl.cm.gray)

ax = plt.subplot(362)
ax.set_title("Pass 1 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(363)
ax.set_title("Pass 2 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(364)
ax.set_title("Pass 3 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass3_Image_Minima, cmap=mpl.cm.gray)


ax = plt.subplot(365)
ax.set_title("Pass 4 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass4_Image_Minima, cmap=mpl.cm.gray)

ax = plt.subplot(366)
ax.set_title("Pass 1,2,3,4 Minima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(min_avg, cmap=mpl.cm.gray)

ax = plt.subplot(367)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image, cmap=mpl.cm.gray)

ax = plt.subplot(368)
ax.set_title("Pass 1 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(369)
ax.set_title("Pass 2 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(3,6,10)
ax.set_title("Pass 3 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass3_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(3,6,11)
ax.set_title("Pass 4 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass4_Image_Maxima, cmap=mpl.cm.gray)

ax = plt.subplot(3,6,12)
ax.set_title("Pass 1,2,3,4 Maxima Image", fontsize=7)
ax.set_axis_off()
ax.imshow(max_avg, cmap=mpl.cm.gray)

ax = plt.subplot(3,6,13)
ax.set_title("Original Image", fontsize=7)
ax.set_axis_off()
ax.imshow(image, cmap=mpl.cm.gray)

ax = plt.subplot(3, 6, 14)
ax.set_title("Pass 1 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass1_image_avg, cmap=mpl.cm.gray)

ax = plt.subplot(3, 6, 15)
ax.set_title("Pass 2 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass2_image_avg, cmap=mpl.cm.gray)


ax = plt.subplot(3, 6, 16)
ax.set_title("Pass 3 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass3_image_avg, cmap=mpl.cm.gray)


ax = plt.subplot(3, 6, 17)
ax.set_title("Pass 4 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(pass4_image_avg, cmap=mpl.cm.gray)

ax = plt.subplot(3, 6, 18)
ax.set_title("Pass 1,2,3,4 Min and Max Image", fontsize=7)
ax.set_axis_off()
ax.imshow(min_max_avg, cmap=mpl.cm.gray)

plt.show()
