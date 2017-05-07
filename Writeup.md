# Behavioral Cloning 

## Writeup

##### Author: Sergey Morozov

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) 
individually and describe how I addressed each point in my implementation.  

---
### Files Submitted and Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [Writeup.md](./Writeup.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) 
and my [drive.py](./drive.py) file, the car can be driven autonomously around 
the track by executing 
```sh
python drive.py model.h5
```

#### 3. Code is usable and readable

The [model.py](./model.py) file contains the code for training and 
saving the convolution neural network. The file shows the pipeline I used 
for training and validating the model, and it contains comments to explain how 
the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've used neural network architecture similar to one, that has been proposed
by Nvidia in the 
[article](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

I've slightly changed the model though. Cropping layer was added that transforms
the image of shape (160, 320) to (70, 320). There is also a normalization layer
(a Keras lambda layer). The number of neurons in fully connected layers
were changed to 120->30->1 instead of 100->50->10. 
The model includes RELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. I just simply collected enough data 
to prevent overfitting. The data collection strategy will be described later in this
writeup.

The model was trained and validated on different data sets to ensure 
that the model was not overfitting. The model was tested by running it 
through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides 
of the road. The data was collected from both tracks. Initially there were
3 center lane driving laps on track 1 and 3 center lane driving laps on track 2.
There were also 1 recovery laps per each track. Later I've added 1 center lane driving laps 
per each track and recovery data from tricky places.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used Nvidia's model architecture because they've successfully applied it for the similar problem.
I've also tried [Comma.ai's model architecture](https://github.com/commaai/research/blob/master/train_steering_model.py)
but it did not perform as good as Nvidia's.

At first I've tried Nvidia's architecture, but with BGR image input. There was a problem with overall stability 
of the driving and with certain turns onthe second half of the first track.

Comma.ai's model performed slightly better with the same input (BGR image), but it still had a 
problem with some edges of the road on the first track. But this was the first model that was able to autonomously drive 
the full lap on the track 1. Unfortunatelly the driving was unsatisfactory.

Then I've collected more data and read the following articles:
- [Vivek Yadav's article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
- [Mengxi Wu's article](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234)
- [Denise R. James' article](https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184)

I've changed an image preprocessing. I used S (saturation) channel from the HSV encoded image. 
Thus, we are getting a gray scale image. 
The contours of the roadway are clearly visible in S channel.

I've slightly changed the number of neurons in dense layers in Nvidia's model. 
Then I started trainig this model again from scratch and after only 1 epoch it was able to 
autonomously drive the car around the track witout even touching lane lines on the edges of the road.

I've collected in total 117630 images with corresponding steering angles and augmented this data by
flipping each image to prevent left or right biases in angles. I got 235260 imges data set in total.
The data archive can be obtain using the [link](https://yadi.sk/d/PaOHVil33HnCKz).

The data set was split to training and validation data set with a 4:1 proportion (i.e 20% of images in validation data set).
Images were also shuffled appropriately.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.
It was also able to drive around track 2 semiautonomously with a few manual interventions. 

On my machine (Lenovo T430s) model can drive with the maximum speed of 20 mph around the track 1 
and with 6 mph around the track 2. My laptop is not able to generate steering angles with the needed frequency 
on the higher speeds.

#### 2. Final Model Architecture

The final model architecture consisted of a Nvidia-like convolution neural network with the following layers and layer sizes.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 70, 320, 1)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 70, 320, 1)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   624         lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 110)           464750      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 30)            3330        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             31          dense_2[0][0]                    
====================================================================================================
Total params: 598,259
Trainable params: 598,259
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set and Training Process

To capture good driving behavior, I recorded laps on track one and two using center lane driving. Then I recorded the vehicle 
recovering from the left side and right sides of the road back to center so that the vehicle would learn to recove from undesired situations.

To augment the data set, I also flipped images and angles thinking that this would mitigate the left angles biases in the dataset.

Visualization of the preprocessing pipeline can be found in [image.ipynb](./image.ipynb). Data augmetation via image flipping
was performed in [data.ipynb](./data.ipynb).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 as evidenced by the driving 
behavior on the track 2 (which is harder). Also, the model started overfitting after the 6th training epoch. 
I used an adam optimizer so that manually training 
the learning rate wasn't necessary.

