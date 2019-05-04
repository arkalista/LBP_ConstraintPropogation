import argparse
import logging
########## Required for Load Image and Display ###########
import imageio as imio
import skimage as imsk
import matplotlib as mpl
import matplotlib.pyplot as plt

########## Required for Generating Variations ###########
from MCAIncludes import *


def generateImageVariations(img):
    imgVar = {}
    # Test Image
    # image = np.array([[20, 27, 24, 26], [16, 23, 30, 32], [22, 22, 20, 19], [22, 10, 35, 19]])
    # print(image.shape)
    # Resize
    imgVar['image'] = imsk.transform.resize(img, (64, 64))

    # Call Genrate Variation from MCAIncludes
    ImgLst = GenerateImageVariations_Minima(imgVar['image'])
    imgVar['pass1_Image_Minima'] = ImgLst[0]
    imgVar['pass2_Image_Minima'] = ImgLst[1]
    imgVar['pass3_Image_Minima'] = ImgLst[2]
    imgVar['pass4_Image_Minima'] = ImgLst[3]

    ImgLst = GenerateImageVariations_Maxima(imgVar['image'])
    imgVar['pass1_Image_Maxima'] = ImgLst[0]
    imgVar['pass2_Image_Maxima'] = ImgLst[1]
    imgVar['pass3_Image_Maxima'] = ImgLst[2]
    imgVar['pass4_Image_Maxima'] = ImgLst[3]

    imgVar['min_avg'] = ((imgVar['pass1_Image_Minima'] + imgVar['pass2_Image_Minima'] + imgVar['pass3_Image_Minima'] +
                          imgVar['pass4_Image_Minima']) / 4)
    imgVar['max_avg'] = ((imgVar['pass1_Image_Maxima'] + imgVar['pass2_Image_Maxima'] + imgVar['pass3_Image_Maxima'] +
                          imgVar['pass4_Image_Maxima']) / 4)

    imgVar['pass1_image_avg'] = ((imgVar['pass1_Image_Minima'] + imgVar['pass1_Image_Maxima']) / 2)
    imgVar['pass2_image_avg'] = ((imgVar['pass2_Image_Minima'] + imgVar['pass2_Image_Maxima']) / 2)
    imgVar['pass3_image_avg'] = ((imgVar['pass3_Image_Minima'] + imgVar['pass3_Image_Maxima']) / 2)
    imgVar['pass4_image_avg'] = ((imgVar['pass4_Image_Minima'] + imgVar['pass4_Image_Maxima']) / 2)

    imgVar['min_max_avg'] = ((imgVar['min_avg'] + imgVar['max_avg']) / 2)

    return imgVar


def plot(_plt, narr, arr, p, q, r):
    ax = _plt.subplot(p, q, r)
    ax.set_title(narr, fontsize=5)
    ax.set_axis_off()
    ax.imshow(arr, cmap=mpl.cm.gray)
    return _plt


def plotImgVar(plt, imname, imgVar, type='all'):
    imgVarType = ['image', 'pass1_Image_Minima', 'pass2_Image_Minima', 'pass3_Image_Minima', 'pass4_Image_Minima',
                  'min_avg', 'pass1_Image_Maxima', 'pass2_Image_Maxima', 'pass3_Image_Maxima', 'pass4_Image_Maxima',
                  'max_avg', 'pass1_image_avg', 'pass2_image_avg', 'pass3_image_avg', 'pass4_image_avg', 'min_max_avg'
                  ]

    plt.figure(1)
    plt.title("Image Varations")

    column = 6

    if type == 'all':

        row = 3
        elements = 16

        for i in range(0, elements):
            plt = plot(plt, imgVarType[i], imgVar[imgVarType[i]], row, column, i + 1)
        plt.savefig(imname + '_all.png', bbox_inches='tight')

    elif type == 'minima':
        row = 1
        elements = 5
        plt = plot(plt, imgVarType[0], imgVar[imgVarType[0]], row, column, 1)
        for i in range(0, elements):
            plt = plot(plt, imgVarType[i + 1], imgVar[imgVarType[i + 1]], row, column, i + 2)
        plt.savefig(imname + '_minima.png', bbox_inches='tight')

    else:
        row = 1
        elements = 5
        plt = plot(plt, imgVarType[0], imgVar[imgVarType[0]], row, column, 1)
        for i in range(0, elements):
            plt = plot(plt, imgVarType[i + 6], imgVar[imgVarType[i + 6]], row, column, i + 2)
        plt.savefig(imname + '_maxima.png', bbox_inches='tight')

    return plt


def loadImage(dir='../assets/lena.png'):
    image = []
    try:
        image = imio.imread(dir, pilmode='F')
    except Exception as e:
        logging.error(str(e))
        exit()

    return image


def parse_cmd_line_args_and_run(plt):
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        help="The type of augmentation that needs to be generated , can take up following values : "
                             "minima, maxima, all")
    parser.add_argument("--input", help="Expecting an input image that is placed in assets folder")

    args = parser.parse_args()

    # check if the input image exists.
    if args.input:
        base_dir = '../assets/'
        img_dir_in = base_dir + args.input
        img_dir_out = base_dir + args.input.split('.')[0]
        image = loadImage(img_dir_in)
    else:
        logging.error('Houston we have a problem !!! - Need image name as input')
        exit()

    # check if user wants to generate all variations

    imgVar = generateImageVariations(image)

    if args.type == 'all':
        plt = plotImgVar(plt, img_dir_out, imgVar, 'all')
        return plt
    elif args.type == 'minima':
        plt = plotImgVar(plt, img_dir_out, imgVar, 'minima')
        return plt
    elif args.type == 'maxima':
        plt = plotImgVar(plt, img_dir_out, imgVar, 'maxima')
        return plt
    else:
        logging.error("Houston we have a problem !!! - Wrong Input for the type of augmentation selected")
        exit()


if __name__ == "__main__":
    plt = parse_cmd_line_args_and_run(plt)
    plt.show()
