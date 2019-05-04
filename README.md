## From Local Binary Patterns to Images

### Introduction:
<p align="center">
  <img width="700" height="500" src="https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lena_all.png">
</p>

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

1. Run the git clone command to copy the contents of folder to your local system, 
<br/> for instance : git clone git@github.com:m-alishan/LBP_ConstraintPropogation.git
2. navigate into the src directory, where you can see the requirements.txt file
<br/> cd LBP_ConstraintPropogation/src
3. Run the following command to install all the dependencies required to run the code
<br/> pip install -r requirements.txt

### Generating Image 

1. To generate the augmented images, place the source image in assets folder, for instance ./assets/lenna.png
2. To generate image variations from local minima, call the following:
    <br/> python Generator.py --type minima --input lenna.png
3. To generate image variations from local maxima, call the following:
    <br/> python Generator.py --type maxima --input lenna.png
4. To generate image variations from both minima and maxima, call the following:
    <br/> python Generator.py --type all --input lenna.png
5. There will be an output image in the assets folder with lenna_minima.png / lenna_maxima.png / lenna_all.png

Note : All the RGB images are converted to GS first and scaled to 64x64, the scaling is done due to performance.

### Sample output on the Lenna Image

Variations Generated from Local Minima Expansion
<p align="center">
  <img src="https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lena_minima.png">
</p>

Variations Generated from Local Maxima Expansion
<p align="center">
  <img src="https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lena_maxima.png">
</p>

### Precomputed Results on CIFAR-10
You can find precomputed results on CIFAR-10 in the CIFAR/DATA directory, directory follows the sructure:
<br/> LBP_ConstraintPropogation/src/CIFAR/DATA/<class_label>/data_batch_<classlabel>_<variation_type>.h5
<br/> For Example
1. Data for class label-0(airplane) and variation type-0(Original Image) 
<br/> LBP_ConstraintPropogation/src/CIFAR/DATA/0/data_batch_0_0.h5
2. Data for class label-1(automobile) and variation type-5(Original Image) 
<br/> LBP_ConstraintPropogation/src/CIFAR/DATA/1/data_batch_1_5.h5

### Acknowledgment:
I'd like to acknoweldge all the contributors who have worked on this repository. My special thanks to Dr.Tahir Syed, Sadaf Behlim, Yameen Malik and Zaid Memon.

If you encounter any issue running the code, or come across somthing you want to add, please do let me know. 
