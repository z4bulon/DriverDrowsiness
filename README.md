# DriverDrowsiness

Dataset is not included in the project and repository, it can be found here: https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new 

Introduction

Problem
	
  Driver drowsiness is a big problem in modern world. With increasing of auto-pilot cars usage, it is very important to keep drivers attention to the road. Artificial Intelligence is still not perfect, so this problem will not be solved by just using auto-pilot. In example of Tesla, AI still needs driver to be close for a steering wheel because if there will be any issue, driver can take control.

Another solutions

Today a lot of companies try to implement different solutions and systems to prevent driver getting drowsy. According to the Wikipedia, different car companies developed their own solutions. For example, Mercedes-Benz developed their ‘Attention Assist’ system back in 2009. This system monitors drivers fatigue level and issues a visual and audible alarm when driver gets drowsy. Volvo Cars implemented the world's first Driver Drowsiness Detection System in 2007. This system is similar to Mercedes-Benz system, so it’s also alerts driver when he gets drowsy.
  
Current work

In this project the program analyzes eyes position to evaluate are eyes opened or closed by using specially trained model which is estimating the level of driver drowsiness. If program decides that driver fall asleep, then program shows a visual alert to the driver. To develop this project I used a different libraries and modules: OpenCV is used to face and eyes detection, also it used to build a special window and webcam usage; Keras uses Tensorflow as a backend and was used to build and train our model.
  
Data and methods

Information about the data

As a data in this project was used a dataset from Serena Raju that was uploaded on Kaggle. This dataset consists of train and test data which includes opened and closed eyes picture. (Picture 1 and 2)

Also, it uses a haar cascade files to detect the face and both eyes that are used in our program. It uses a Viola-Jones algorithm and was used with OpenCV library to detect the face and eyes and estimate is that a face and eyes or not. There is used not an original OpenCV dataset, it uses an alternative dataset to train. The dataset was downloaded from OpenCV official GitHub. (Picture 3)
 
Description of the ML model

Basically, our model is used to recognize whether is person eyes are open or closed. We used a keras to model. Model uses Convolutional Neural Networks (CNN). CNN is very useful in image classification. It is a type of deep neural network that performs very accurately and efficiently. As we know, CNN consists from 3 type of layers: hidden, input and output. Our model is performed on hidden layers and uses a 2D matrix multiplication on the layers. In this CNN we used 3 hidden layers with convolutional layer, 64 nodes and 3 kernel size and 1 fully connected layer with 128 nodes. All of this layers uses Relu activation. The output layer has 2 nodes and uses Softmax activation. The code of model is pinned in archive. Also, before constructing a model, I made a path where dataset is stored, so it can use the dataset to train the model.
  
Results
 
As you can see, when program starts we can see a new window which includes the input from webcam. The face of user is shaped in square and movement of this shape is directly depends on the movement of user’s face or head. Also it has a message where user can see how much seconds his eyes where closed. 
 
On the Picture 5 is shown that if users eyes are closed, this windows borders will be painted in white and there will be a message that says: “Stop the car”. The program analyze my eyes movement and if it see that my eyes are closed more than 15 seconds, it will show this alert. 
 
As you can see above, accuracy of the model was improved from 0.75 up to 0.96. 
The train dataset includes 1234 pictures of open and closed eyes. 

Link to the video sample: https://youtu.be/d4Fut39yGj4
