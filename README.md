# SeRanet
Super Resolution of picture images using deep learning
SeRanet project aims to study, research and develop to improve super resolution quality


## Introduction
SeRanet upscales picture image size to x2.

Before upscale

![input picture1](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/3/photo3_xinput.jpg)
![input picture2](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/4/photo4_xinput.jpg)
![input picture3](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/1/photo1_xinput.jpg)

Upscaled image using lanczos method (with OpenCV library)

![lanczos picture1](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/3/lanczos.jpg)
![lanczos picture2](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/4/lanczos.jpg)
![lanczos picture3](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/1/lanczos.jpg)

**Upscaled image using SeRanet**

![seranet_v1 picture1](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/3/seranet_v1.jpg)
![seranet_v1 picture2](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/4/seranet_v1.jpg)
![seranet_v1 picture3](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/1/seranet_v1.jpg)

Original image (= Ground truth data, for reference)

![original picture1](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/3/photo3_original.jpg)
![original picture2](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/4/photo4_original.jpg)
![original picture3](https://raw.githubusercontent.com/corochann/SeRanet/master/assets/compare/1/photo1_original.jpg)

## Description

It is developed with python on chainer framework, flexible machine learning library.

SeRanet project aims to Study and Research how deep convolutional neural network works
to learn super resolution of the image.


## References
The project is inspired by following two reference

 - Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, "Image Super-Resolution Using Deep Convolutional Networks",
 [http://arxiv.org/abs/1501.00092](http://arxiv.org/abs/1501.00092)

 The term "SRCNN", super resolution using deep convolutional neural network, is introduced in this paper.

 - [waifu2x](https://github.com/nagadomi/waifu2x)

 It is the popular project for image super resolution for Anime-Style art.

Machine learning library
 - [chainer](http://chainer.org/)

 Machine learning library which can be written in python.
 It is open source on [github](https://github.com/pfnet/chainer)

## How to use

### Basic usage
Just specify image file path which you want to upscale.

Ex. Upscaling input.jpg
```
python src/inference.py input.jpg
```

### Specify output file name and path
Ex. Upscaling /path/to/input.jpg to /path/to/output.jpg
```
python src/inference.py /path/to/input.jpg /path/to/output.jpg
```

### Specify model to use:
SeRanet project studies several network architecture.
You can specify which network archtecture to use for SR with -a option.

Ex. use model seranet_v1
```
python src/inference.py /path/to/input.jpg /path/to/output.jpg -a seranet_v1
```

### Use GPU:
GPU makes calculation much faster.
Specify -g option is to use GPU.

```
python src/inference.py /path/to/input.jpg /path/to/output.jpg -g 0
```



## Training

You can construct your own convolutional neural network, and train it!

###  1. Data preparation
Put training images[1] inside data/training_images directory.
(I used 5000 photo images during the training.)

[1]: Currently, images will be cropped to size 232 x 232 during training.

<!---
###  2. Construct your model/arch (convolutional neural network, etc)
(Skip this procedure if you want to train existing model)
You can create your own arch in code/arch/ directory.
Please refer other arch implementations to design your own neural network.

###  3. Training the model
Once prepared training_images and model to be trained, you can train your model by
```
python src/train.py -a model_name
```
-->

###  2. Training the model
Once prepared training_images, see code/arch/ directory to choose which model to train, and execute below.
-g 0 is to use GPU.
(For the training, it is highly recommended to use GPU, otherwise training don't finish maybe a month...)
```
python src/train.py -a model_name -g 0
```

## Contribution is welcome

The performance of SR for this project is not matured.
You are welcome to suggest any improvement & contribute to this project.
If you could get any model which performs better performance, feel free to send me a pull request!


