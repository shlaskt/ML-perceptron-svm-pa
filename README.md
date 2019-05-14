# ML-perceptron-svm-pa
1. [Introduction](#introduction)  
2. [Dataset:](#dataset)  
3. [Code:](#code)  
4. [Note:](#note)


## Introduction
In this exercise we were implement and compare Perceptron, SVM, and PA. We were provided with
a training set of 3,286 examples where we needed to classify between 3 classes.

## Dataset:
Our data set is about Abalone age classification (https://en.wikipedia.org/wiki/Abalone).
In this dataset, we were provided with eight features per instance -seven of which are numerical and one is the categorial, and three labels.
Labels correspond to the Abalones age. We should explored different ways to convert this categorial attribute to a numerical one.
Moreover, to try different normalization techniques as well as feature selection.

## Code:
Our main file called: ex2.py
Our code should get as input three arguments. 
* The first one will be the training examples (train_x.txt).
* The second one is the training labels (train_y.txt).
* The third one will be the testing examples (test_x.txt).
train_x.txt and test_x.txt will have the same format.
We trained 3 algorithms: Perceptron, SVM and PA (in that order).
Than, we output to the screen our predictions (only) for test_x.txt in the following format:
* perceptron: 0, svm: 0, pa: 1
* perceptron: 2, svm: 2, pa: 2
* perceptron: 1, svm: 1, pa: 1

...

where each line in the output corresponds to a line in test_x.txt and the numbers represent our class predictions for classes 0,1,2.

## Note:
The code should run up to 5 minutes long.
