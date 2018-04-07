# Convolutional Neural Networks for recognizing dots on dice

##### This README file will be updated to match with both teoretical part and practital achievements in this repository

### Technologies used:
* Python3
* OpenCV
* Jupyter Notebook
* Numpy
* Scikit learn
* matplotlib
* Tensorflow
* Keras
* AWS EC2 with bitfusion
* Nvidia CUDA
* Google Compute Engine
* Julia + Knet.jl

### Goals
Achieve best possible recognition of dots on each dice wall. This neural model will be used in camera, 
now presented [here](https://github.com/oziomek1/neural_network_dice/blob/master/camera_with_neural_network_test.ipynb) 
in a notebook. Unfortunately, real-time update of matplotlib plot 
is rather anoying, now this feature is moved [there](https://github.com/oziomek1/neural_network_dice/blob/master/real_time_camera_neural_network.py)
to a python script. This works in separate, new window with real-time update.

### Datasets
* First 120 real images, 20 images per one side of dice. This is multiplicated to 60480 images due to 
warping perspectives and rotations in openCV-s. 

Code for this is available [here](https://github.com/oziomek1/image_database)
This is not this kind-of *beautiful code*, but it's useful for me and I hope I'll have some time later to refactor it

* Next-tier dataset is 600 images with 100 for each side of a dice with 15 different color settings for dots, dices and background.
This is transformed to 100800 dices with 64x64x3 or 64x64x1 sized, depends on particular network model.

* Last dataset is 984 images, 164 per wall. This time, dices are not focused in the center of a background which lets me 
now use trainings with rectangular images with aspect ratio 4:3, the same as photos taken from camera.
Now this set is available in 640x480x1, 320x240x3, 320x240x1, 160x120x1 and 106x79x1

Last pre-trained model are using the smallest version 106x79x1, due to horrendous computation time, event on AWS Nvidia K80 TESLA GPU

### Achievements
* Trained models with accuracy around 99% for 64x64x1 (grayscale) images with set colors of both background and dice
* Models with accuracy over 98% for 64x64x1 (grayscale) and 64x64x3 (RGB) images with 15 different sets of colors.
* First model with accuracy of 68.03% for 106x79x1 (grayscale) finally breaking the 64x64 initial size od learned images

### Training
All neural network models are trained on AWS EC2 and  Google Compute Engine (so far mainly AWS)
On AWS it's using p2.xlarge instance with Nvidia TESLA K80 GPU

### Troubles with Keras
* Different numbers of parameters in compiled models for both Sequential and Functional API model
* Troubles with setting proper batch_size according to each images resolutions (number of parameters to store in GPU memory)
* Almost not-learnable images with recognition greater than 64x64. The only success so far is with 106x79 images (68.03% accuracy). 
With 160x120 the error/accuracy rate remains unchanged (marginal upgrades, can be treated as noise) 

### Frameworks
As for now, almost all notebooks are based on Keras framework with Tensorflow backend, due to simplicity at the beginning.

The future will focus on changing tech-stack towards Tensorflow. More ambitious plans is to leave TF ans Keras and 
create CNN from scratch, purely in C++ with further application of Nvidia CUDA support for speed.

<img src="/photos/kolaz.png">
##### updating README, to be continued...

## Below some models visualizations:

#### These below are to exactly the same neural networks, one flattened and another one, deeper with substitutions in convolutional filters for faster computing 
<img src="/model_plots/NON-substituting_160x120_plot.png" style="width: 400px;"/>
<img src="/model_plots/substituting_160x120_plot.png" style="width: 400px;"/>


#### Model with Adam optimizer:
<img src="https://raw.githubusercontent.com/oziomek1/neural_network_dice/master/model_plots/model2adam_plot.png">

#### Model with Adam optimizer but this is a functional model, not Sequential model:
<img src="https://raw.githubusercontent.com/oziomek1/neural_network_dice/master/model_plots/model2API_plot.png">
