# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_Images/traindataset_visualization.jpg "Visualization"
[image2]: ./Output_Images/grayscale.jpg "Grayscaling"
[image3]: ./Output_Images/randomrotation.jpg "Random Rotation"
[image4]: ./Output_Images/newdataset_visualization.jpg "New Dataset Visualization"
[image5]: ./Output_Images/Mytestimages.jpg "Traffic Sign "
[image6]: ./Output_Images/image_0_predictions.jpg "Traffic Sign 1"
[image7]: ./Output_Images/image_1_predictions.jpg "Traffic Sign 2"
[image8]: ./Output_Images/image_2_predictions.jpg "Traffic Sign 3"
[image9]: ./Output_Images/image_3_predictions.jpg "Traffic Sign 4"
[image10]: ./Output_Images/image_4_predictions.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing data distribution

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun showed their model achieved higher accuracy by using several techniques which included ignoring color information. They explained that "the wide array lighting conditions make color in general unreliable.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As grayscaling, I normalized the image data because standardizing the inputs can make training faster and reduce the chances of getting stuck in local optima. 

I decided to generate additional data because the original dataset is skewed. Some labels have 10 times of numbers of data than others.

To add more data to the data set, I used the techniques such as random translation, rotation and scale of the original image. I used these techniques because the neural network I trained should be insensitive to feature positions.

Here is an example of an original image and an augmented image by using random rotation

![alt text][image3]

I increased the size of training dataset to 80933. Here is bar plot showing the augmented data distribution

![alt text][image4]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        									| 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 					|
| RELU					|																|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 								|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 					|
| RELU					|																|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 									|
| Flatten         		| outputs 1600  												| 
| Fully connected		| outputs 600        											|
| RELU					|																|
| Dropout 				| 0.65 keep        												|
| Fully connected		| outputs 240        											|
| RELU					|																|
| Dropout 				| 0.65 keep        												|
| Fully connected		| outputs 43        											|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with 
* batch size : 128
* epochs : 60
* learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.963 
* test set accuracy of 0.949


If a well known architecture was chosen:
* I chose LeNet-5
* I chose LeNet-5 implementation from the lecture because it gave high accuracy on the prediction of MNIST dataset. In addition, I increased the number of weights for each layer because I assume traffic signs have more features to capture. To prevent overfitting, I also added dropout for the fully connected layers (except the output layer).
* We can tell that the model is not underfitting because of the high training accuracy. The model is not overfitting (maybe a little but acceptable) because I also get high accuracy on the validation and test set. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 

* The first image might be difficult to classify because the edge features are the same with other signs like "Beware of ice/snow", "Slippery road","Wild animals crossing" and so on. The plane features (with low resolution) are close to those signs, too. 

* The second image might be easy to classify because it looks distiguishable from other signs.

* The third image might be difficult to classify because it is quite simlar with the sign " Speed limit (20km/h)".

* The fourth image might be easy to classify because the features it has make it easy to separate from other signs.

* The fifth image might be difficult to classify. The reason is similar with that for the first image.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Right-of-way  at the next intersection 	| Right-of-way  at the next intersection  		| 
| Turn left ahead     						| Turn left ahead 								|
| Speed limit (70km/h)						| Speed limit (70km/h)							|
| Roundabout mandatory	      				| Roundabout mandatory					 		|
| Road work									| Road work     								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Top 5 softmax probabilities for each image: 

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

