## From Local Binary Patterns to Images

### Introduction:

![All Variations](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/AllVariations.png)

Data augmentation techniques have been employed to overcome the problem of model over-fitting in deep convolutional neural networks and have consistently shown improvement in classification. Most data augmentation techniques perform affine transformations on the image domain. However these techniques cannot be used when object position is significant in the image. In this work we propose a data augmentation technique based on sampling a representation built by inequality constraints propagated from local binary patterns. 

### Citations

If you are using the code provided here in a publication, please cite our paper:

  @inproceedings{syed2015leveraging,
    title={Leveraging mutual information in local descriptions: from local binary patterns to the image},
    author={Syed, Tahir Q and Behlim, Sadaf I and Merchant, Alishan K and Thomas, Alexis and Khan, Furqan M},
    booktitle={International Conference on Image Analysis and Processing},
    pages={239--251},
    year={2015},
    organization={Springer}
  }
  
### Installing 


### Generating Image 

Variations Generated from Local Minima Expansion
![Local Minima Expansion](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lenna_MinimaVariations.png)

Variations Generated from Local Maxima Expansion
![Local Maxima Expansion](https://raw.githubusercontent.com/arkalista/LBP_ConstraintPropogation/master/assets/lenna_MaximaVariations.png)


### Precomputed Results on CIFAR-10
If you want to compare your method with HED and need the precomputed results, you can download them from (http://vcl.ucsd.edu/hed/eval_results.tar).


### Acknowledgment: 


If you encounter any issue running the code, or come across somthing you want to add, please do let me know. 
