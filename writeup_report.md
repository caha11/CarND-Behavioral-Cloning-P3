# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)


[image1]: ./examples/cnn-architecture-624x890.png "Nvidia's Car NN"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model  
* drive.py for driving the car in autonomous mode  
* model.h5 containing a trained convolution neural network  
* writeup_report.md summarizing the results
* run1.mp4 video of the track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model was taken from the Nvidia's paper on a self driving car neural network. Source: <https://devblogs.nvidia.com/deep-learning-self-driving-cars/>

![alt text][image1]

Hence the model consisted of:

* Convolutional Layer - filters: 24 - kernel size: 5x5 - strides: 2 - Activation: RELU
* Convolutional Layer - filters: 36 - kernel size: 5x5 - strides: 2 - Activation: RELU
* Convolutional Layer - filters: 48 - kernel size: 5x5 - strides: 2 - Activation: RELU
* Convolutional Layer - filters: 64 - kernel size: 3x3 - strides: 1 - Activation: RELU
* Convolutional Layer - filters: 64 - kernel size: 3x3 - strides: 1 - Activation: RELU
* Fully Connected Layer - 1164
* Fully Connected Layer - 100
* Fully Connected Layer - 50
* Fully Connected Layer - 10
* Fully Connected Layer - 1

Normalisation was done before the first Convolutional layer using the Keras' Lambda function (model.py - line 110)

#### 2. Attempts to reduce overfitting in the model

To help the model generalise, additional data was generated. Flipped images, as well as right and left images from the cameras were added. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I attempted to gather the data using the remote simulator, however it was extremely laggy and hard to control. I then tried to gather the data using the simulator on the local machine. However, uploading the data to the online workspace was taking VERY long, and the workspace would refresh mid-upload.

Hence, I had to resort to Udacity's sample data. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first I tried the LeNet model which was working fine for the first few seconds, but it wasn't sufficient for the full lap. The model was too small for this system. More layers could be added to combat the underfitting, but I decided to go for a more proven model from Nvidia which had 5 convolutional layers compared to LeNet's 2.


#### 2. Final Model Architecture

The final model architecture (model.py lines 108-125) is discussed in "Model Architecture and Training Strategy" (point 1.)

#### 3. Creation of the Training Set & Training Process

To recover from going too far to the left and too far to the right, the left and right images from the left and right cameras were used. The corresponding measurements were then added with a correction factor of 0.2 (left: +0.2, right: -0.2).

To prevent overfitting and left turn bias, the images with corresponding measurements were flipped and added to the training and valid data.


After the collection process, I had ~120k number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2, since the accuracy wasn't improving that greatly after the 1st epoch. Also the training accuracy and validation accuracy were pretty close after the 2nd epoch.
