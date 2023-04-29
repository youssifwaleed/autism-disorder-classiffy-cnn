# Autism Disorder Classiffication in CNN

## Introduction
This code was a part of a competition for CS students in Neural Network and Deep Learning course at FCIS. 
The goal of the competition was to build deep learning models that can diagnose autism disorder in the children from their face. 

## DataSset
The proposed model is a CNN model constructed of 5 layers that runs on a dataset as provided in the competition profile in Kaggle 
```kaggle competitions download -c fcis-asu-autism-disorder-classification``` , the dataset is compsed of 2 folders (training dataset,testing dataet).
## Model
The proposed model is CNN model composed of 5 Layers, but before the model a preprocessing step was made to the dataset to shuffle the dataset and ease the process of modeling.
By compining both training dataset folders of autistic and non-autistic, diving them into 70% training set and 30% testing set.
Then the model is a sequential model composed of 5 Layers each layer has goes through the Convolutional Layer and after that an activation function and for this model we choose MaxPooling.
