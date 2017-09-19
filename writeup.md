#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/distribution.png "Distribution"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Test_images_from_net/1.jpg "Traffic Sign 1"
[image5]: ./Test_images_from_net/2.jpg "Traffic Sign 2"
[image6]: ./Test_images_from_net/3.jpg "Traffic Sign 3"
[image7]: ./Test_images_from_net/4.jpg "Traffic Sign 4"
[image8]: ./Test_images_from_net/5.jpg "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zishanahmed08/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the different classes.Ideally a dataset should have an equal representation of all classes.So that it does not bias towards one particular class.
Unfortunately the dataset is not well distributed. Class 2: Speed limit (50km/h)  has around 2010 samples whereas Class 19: Dangerous curve to the left  has just 180 samples.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun mentioned in their paper, using color channels didn’t seem to improve things a lot, so I will only take Y channel of the YCbCr representation of an image.

As a last step, I normalized the image data because it reduces the gradient vanishing problem, as it is easier to initialize the weights relative to the input strength, and allows to use higher learning
rates.
Reference : Ioffe and Szegedy (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv:1502.03167


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented the Sermanet/LeCun model from their traffic sign classifier paper and saw good results. The paper doesn't go into detail describing exactly how the model is implemented (particularly the depth of the layers).
I began with Truncated Normal intialisation(mu = 0,sigma = 0.1) and then switched to Xavier initializer, which automatically determines the scale of initialization based on the layer's dimensions leaving me with lesser hyper parameters to tweak.
My final model consisted of the following layers:

1.5x5 convolution (32x32x1 in, 28x28x6 out)
2.ReLU
3.2x2 max pool (28x28x6 in, 14x14x6 out)
4.5x5 convolution (14x14x6 in, 10x10x16 out)
5.ReLU
6.2x2 max pool (10x10x16 in, 5x5x16 out)
7.5x5 convolution (5x5x6 in, 1x1x400 out)
8.ReLu
9.Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
10.Concatenate flattened layers to a single size-800 layer
11.Dropout layer
12.Fully connected layer (800 in, 43 out)
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer as suggested in the lessons.The hyperparameters like dropout probabibilty (0.5) and learning rate (0.0001) were pretty much the standard hyperparameters suggested in the lesson.
The dropout probabilty was in range of the [original paper by srivastava](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) .The number of epochs was set as per the computational resources

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I sticked to the Sermanet Le Cun model as it was proved to work.The hyperparameters where pretty much frozen at optimal values.At the second attempt i switched  Truncated normal intialisation with Xavier intialisation as it reduces the number of hyperparameters to worry about and also recommended by some blogs/tutorials.
For the final model i increased the number of epochs from 60  to 80.However this did not help as Validation accuracy had plateued by 55the epoch and did not improve further.

Log
1st attempt - 92.3%
Intialisation: Truncated normal
batch size: 100, epochs: 60, rate: 0.001,dropout keep prob:0.5, mu: 0, sigma: 0.1

Final attempt - 93.6%
Intialisation: Xavier
batch size: 100, epochs: 80, rate: 0.001,dropout keep prob:0.5

My final model results were:
* validation set accuracy of 93.6%
Code: In [12]
* test set accuracy of 92.1%
Code: In [13]
* deployment set accuracy of 100%
Code: In [15]

If a well known architecture was chosen:
* What architecture was chosen?
Answer: Sermanet/LeCun model from their traffic sign classifier paper.
* Why did you believe it would be relevant to the traffic sign application?
Answer: it was trained with the sole purpose of doing well on GTSRB dataset.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Answer:Validation and Test accuracy is around 93% and the deployment accuracy is  100%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web from [this pdf]http://www.adcidl.com/pdf/Germany-Road-Traffic-Signs.pdf

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        			|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 ton prohibited  | Vehicles over 3.5 ton prohibited   			| 
| Yield     						| Yield										    |
| Turn right ahead					| Turn right ahead								|
| Stop	      						| Stop					 				        |
| Pedestrians						| Pedestrians     							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. All the images are correctly classified with 100% certainty as the images are pretty easy.They are not affected by shadow or other occlusions.Nor do they have poor lighting conditions.
However tougher test images wouldn't be correctly classified.Data augmentation techiniques would help with regularization and histogram equalization would help with feauture extraction.
This compares better than the accuracy on the test set of 92.1%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16,17th cell of the Ipython notebook.

For all the images, the model is completely  sure with probability of 1.0. The top five soft max probabilities are not so useful here.






