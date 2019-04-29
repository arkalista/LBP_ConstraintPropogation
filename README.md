## From Local Binary Patterns to Images

### Introduction:

![All Variations](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/AllVariations.png)

Data augmentation techniques have been employed to overcome the problem of model over-fitting in deep convolutional neural networks and have consistently shown improvement in classification. Most data augmentation techniques perform affine transformations on the image domain. However these techniques cannot be used when object position is significant in the image. In this work we propose a data augmentation technique based on sampling a representation built by inequality constraints propagated from local binary patterns. 

### Citations

If you are using the code provided here in a publication, please cite our paper:

    @inproceedings{merchant2018appearance,
      title={Appearance-based data augmentation for image datasets using contrast preserving sampling},
      author={Merchant, Alishan and Syed, Tahir and Khan, Behraj and Munir, Rumaisah},
      booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
      pages={1235--1240},
      year={2018},
      organization={IEEE}
    }
  
### Installing 


### Generating Image 

1. To generate the augmented images, place the source image in assets folder, for instance ./assets/lenna.png
2. To generate image variations from local minima, call the following:
    python Generator.py --type minima --input lenna.png
3. To generate image variations from local maxima, call the following:
    python Generator.py --type maxima --input lenna.png
4. To generate image variations from both minima and maxima, call the following:
    python Generator.py --type all --input lenna.png
5. There will be an output image in the assets folder with lenna_minima.png / lenna_maxima.png / lenna_all.png

Note : All the RGB images are converted to GS first and scaled to 64x64, the scaling is done due to performance.

### Sample output on the Lenna Image

Variations Generated from Local Minima Expansion
![Local Minima Expansion](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lenna_MinimaVariations.png)

Variations Generated from Local Maxima Expansion
![Local Maxima Expansion](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lenna_MaximaVariations.png)

### Precomputed Results on CIFAR-10


### Acknowledgment:
I'd like to acknoweldge all the contributors who have worked on this repository from time to time, my special thanks to Prof Tahir Syed for his contribution to the idea and Sadaf Behlim, Alexis Thomas, Yameen Malik & Zaid Memon for their contribution in helping me write this code. 

If you encounter any issue running the code, or come across somthing you want to add, please do let me know. 
