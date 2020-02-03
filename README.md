# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goal of this project is train a deep learning model to recognize traffic signs. The steps of this project are:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images



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


---


### Data Exploration

#### 1. Data summary
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing data distribution

![Data distribution][image1]

### Design and Test a Model Architecture

#### 1. Data preprocessing

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun showed their model achieved higher accuracy by using several techniques which included ignoring color information. They explained that "the wide array lighting conditions make color in general unreliable.

Here is an example of a traffic sign image before and after grayscaling.

![transfer to grayscale][image2]

As grayscaling, I normalized the image data because standardizing the inputs can make training faster and reduce the chances of getting stuck in local optima.

I decided to generate additional data because the original dataset is skewed. Some labels have 10 times of numbers of data than others.

To add more data to the data set, I used the techniques such as random translation, rotation and scale of the original image. I used these techniques because the neural network I trained should be insensitive to feature positions.

Here is an example of an original image and an augmented image by using random rotation

![Data augmentation][image3]

I increased the size of training dataset to 80933. Here is bar plot showing the augmented data distribution

![Improved data distribution][image4]




#### 2. Model architecture

I chose the LeNet-5 implementation as the base model  because it gave high accuracy on the prediction of MNIST dataset. In addition, I increased the number of weights for each layer because I assume traffic signs have more features to capture. To prevent overfitting, I also added dropout for the fully connected layers (except the output layer).

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


#### 3. Hyperprarameters

To train the model, I used an AdamOptimizer with
* batch size : 128
* epochs : 60
* learning rate: 0.001

#### 4. Model performance

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.963
* test set accuracy of 0.949

We can tell that the model is not underfitting because of the high training accuracy. The model is not overfitting (maybe a little but acceptable) because I also get high accuracy on the validation and test set.


### Test the Model on New Images

#### 1. Choose new test images

Here are five German traffic signs that I found on the web:

![New test images][image5]

* The first image might be difficult to classify because the edge features are the same with other signs like "Beware of ice/snow", "Slippery road", "Wild animals crossing" and so on. The plane features (with low resolution) are close to those signs, too.

* The second image might be easy to classify because it looks distinguishable from other signs.

* The third image might be difficult to classify because it is quite similar to the sign " Speed limit (20km/h)".

* The fourth image might be easy to classify because the features it has make it easy to separate from other signs.

* The fifth image might be difficult to classify. The reason is similar with that for the first image.



#### 2. Model performance on these new traffic signs

Here are the results of the prediction:

| Image			        					|     Prediction	        					|
|:-----------------------------------------:|:---------------------------------------------:|
| Right-of-way  at the next intersection 	| Right-of-way  at the next intersection  		|
| Turn left ahead     						| Turn left ahead 								|
| Speed limit (70km/h)						| Speed limit (70km/h)							|
| Roundabout mandatory	      				| Roundabout mandatory					 		|
| Road work									| Road work     								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

#### 3. Softmax probabilities for each test images

Top 5 softmax probabilities for each image:

![New test image 1][image6] ![New test image 2][image7] ![New test image 3][image8]
![New test image 4][image9] ![New test image 5][image10]

### Acknowledgement
This project is provided by Udacity
