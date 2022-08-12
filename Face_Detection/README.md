# Face Detection
## Table of contents
* [Overview](#Overview)
* [Tools and Frameworks](#Tools-and-Frameworks)
* [Resources](#Resources)
* [Installations](#Installations)
* [Demonstration](#Demonstration)
* [Notes](#Notes)
* [Repository status](#Repository-status)

## Overview
* The purpose of this repository is ilustrating simple face detection, using Viola-Jones Object Detection Framework.
* The consepts are used also includes:
    * Integral images
    * Haar-Like features
    * AdaBoost
    * Classifier cascades

## Tools and Frameworks
* Opencv lib - Open Source Computer Vision library, which uses Viola-Jones Object Detection Framework for face detection

## Resources
* Real Python - [Tutorial - Traditional Face Detection With Python](https://realpython.com/traditional-face-detection-python/#reader-comments)
* Real Python - [Course - Traditional Face Detection With Python](https://realpython.com/courses/traditional-face-detection-python/)
* CSV repository - [Classifier Cascade for face detection](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml)

## Installations
Please follow the installation instruction in the Readme file of the reposiroty (Machine_Learning/Readme) before the instructions in this section.

* Create a folder for your project
* Open Visual code in the project destination
* Create an enviroment: `conda create --name env_face_detection python=3.5`
* activate the enviroment: `conda activate env_face_detection`
* install the following libraries:
`conda install scikit-learn`
`conda install -c conda-forge scikit-image`
`conda install -c menpo opencv3`
  
* Change to python version which VC will use using the buttom left button and make sure you choose python 3.5



## Demonstration
<kbd>
  <img style="display: block;margin-left: auto;margin-right: auto; width: 100%; height: 100%;" src="https://github.com/gy-m/Machine_Learning/blob/main/Face_Detection/Documentation/Ilustration.png">
</kbd>

## Notes
None

## Repository-status
* Status - Beta
* TODOs - None