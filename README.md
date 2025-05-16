# Cat-Dog Classifier
A classic image classification project that distinguishes cats from dogs using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier.

## Overview
This project loads images of cats and dogs, preprocesses them by resizing and converting to grayscale, and then extracts HOG features to capture important shape and texture information. Using these features, an SVM with an RBF kernel is trained to classify the images. The project evaluates the model's accuracy and visualizes sample predictions.
The dataset used is the popular Cats vs Dogs dataset from Kaggle, which contains thousands of labeled images of cats and dogs.

## Features
Image preprocessing: resizing, grayscale conversion, normalization
Feature extraction with HOG (Histogram of Oriented Gradients)
SVM classifier with RBF kernel for binary classification
Performance evaluation with accuracy score and confusion matrix
Visualization of test predictions for quick validation

## Dataset
The dataset contains thousands of images of cats and dogs with labels indicated by filenames (cat.xxx.jpg or dog.xxx.jpg). This project uses a subset of the dataset for training and testing.
Sample Results
Insert accuracy/confusion matrix image here
Insert sample prediction images here

## Sample outputs
### Confusion Matrix:
![Image Alt](https://github.com/KrishnaMohan2808/Cat-Dog-classifier-using-SVM/blob/de87e5a53df6c66f7fed7b798f8578ec2bc29a80/Screenshot%202025-05-16%20232310.png)

### Sample Predictions:
![Image Alt](

## Requirements
Python 3.x
OpenCV (opencv-python)
scikit-learn
scikit-image
NumPy
Matplotlib

## Install dependencies with:
bashpip install opencv-python scikit-learn scikit-image numpy matplotlib

## Usage
Download and extract the Kaggle Cats vs Dogs dataset into a folder named train in the same directory as this script.
Ensure image filenames contain the label (cat or dog).
Run the classifier script by opening a terminal in the project folder and typing:

bashpython cat_dog_classifier.py

The program will print accuracy and confusion matrix results in the terminal and display sample prediction images.

## How It Works
Loads and preprocesses images (resize to 64x64, grayscale, normalize)
Extracts HOG features from each image
Splits data into training and test sets
Trains an SVM classifier on training data
Predicts and evaluates on test data
Visualizes some predictions for inspection

## Future Scope
Deep Learning: Implement convolutional neural networks (CNNs) for improved accuracy.
Data Augmentation: Add augmentation techniques (rotation, flipping, zooming) to enhance generalization.
Hyperparameter Tuning: Use grid search or random search to optimize SVM parameters.
Real-Time Classification: Integrate with live camera feed for real-time cat-dog detection.
Multi-class Classification: Extend to classify more animal species.


## About
This project demonstrates how classical computer vision and machine learning techniques can effectively classify images with relatively low computational cost. It serves as a great introduction to feature extraction and traditional ML methods in image classification.

## License
This project is licensed under the MIT License.

Feel free to contribute or suggest improvements!
