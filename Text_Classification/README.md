# Text Classification
## Table of contents
* [Overview](#Overview)
* [Tools and Frameworks](#Tools-and-Frameworks)
* [Resources](#Resources)
* [Installations](#Installations)
* [Demonstration](#Demonstration)
* [Notes](#Notes)
* [Repository status](#Repository-status)

## Overview
* The purpose of this repository is to make a "text classification" using deep nural networks. It consists of the following parts:
* Part 1 - For a given text, finding it's "Feature Vector" using BOW model (Bag of Words). The process is:
    * Create a Corpus - A list with sentences as the elements
    * Create a Vocabulary - A dictionary which is based on the corupus and created using CountVectorizer function (a part of scikit-learn library). CountVectorizer returns an object where one of it's properties is the vocabulary.
    * Create a Feature Vector - A vector which is based on the vocabulary, and created using CountVectorizer's object. The method which returns the feature vector is transform. 
* Part 2 - For a given feature vecotr, doing text classification using a deep neural networks model named "Sequential model". The process is:
    * Corpus
        * Create a data file (df) which is a dataframe object (pandas basic object) based on 3 d ifferent sources.
        * Creating 2 feature vectors for:
            * Training set
            * Testing set
        * Create a LogisticRegression model for a "Baseline"
        * Create and train a deep neural networks model
            * Create a Sequential model (a type of keras model)
            * Adding a dense layer type for the inputs. The activation function will be defined as relu.
            *  Adding a dense layer type for the outputs. The activation function will be defined as sigmoid.
            * Specify the loss method, of type 'binary_crossentropy' and a optimizer method of type'adam'. These is done using the 'compile' method (of mode Sequential model)
            * Training, using the 'fit' method (of mode Sequential model)
    

## Tools and Frameworks
* Part 1:
    * scikit-learn lib - Used for:
        * Creation of Vocabulary - CountVectorizer function takes data set and returns an object, where it's "vocabulary_" property is the vocabulary of the corpus.
        * Creation of the Feature Vector - CountVectorizer object have a "transform" method which reutnr a feature vector.

* Part 2:
    * Pandas - For creation of the data set
    * scikit-learn lib
        * train_test_split function - For spliting the data sets (Training and Testing data sets)
        * CountVectorizer function - For creation of the vocabulary and the feature vectors.
        * LogisticRegression model - For Text classification.
    * Keras
        * Sequential model - Text classification
        * Layers
    * matplotlib - For demonstrational purposes.


## Resources
* Real Python - [Tutorial - Practical Text Classification With Python and Keras](https://realpython.com/python-keras-text-classification/#defining-a-baseline-model). The repository relies on this tutorial.
* Real Python - [Course - Learn Text Classification With Python and Keras](https://realpython.com/courses/text-classification-with-keras/)
* Data Set source - [Sentiment Labelled Sentences Data Set Additional Resources](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

## Installations
Please follow the installation instruction in the Readme file of the reposiroty (Machine_Learning/Readme) before the instructions in this section.

* Create a folder for your project
* Open Visual code in the project destination
* Create an enviroment: `conda create name –-Text_Classification python=3.6.13`
* activate the enviroment: `conda activate Text_Classification`
* install the following libraries: 
`conda install scikit-learn==0.24.2`
`conda install numpy==1.19.2`
`conda install keras==2.3.1`
`conda install pandas=1.1.5`


## Demonstration
NTA

## Notes
The code and the documentaion does not meet the pythonic standards due to the will document it in a clear way for unprofessional data sciences programmers. 

## Repository-status
* Status - Beta
* TODOs - Complete the bellow sections from the above tutorial:
    * Word Embedding
    * Convolutional Neural Networks (CNN)