# Edge Detection
## Table of contents
* [Overview](#Overview)
* [Tools and Frameworks](#Tools-and-Frameworks)
* [Installations](#Installations)
* [Demonstration](#Demonstration)
* [Notes](#Notes)
* [Repository status](#Repository-status)

## Overview
* The purpose of this repository is to create an "Edge Detection" profiling using CPU and GPU.
* Soble edge detector is an implementaion of "Edge Detection" alghorithem.
* The fact there are a significant amount of pixels to multiply and summarize, due to the charctericties of this alghorithem, this calculation is complex for the CPU alone, which workd in a procedural method. GPU, unlike CPU, consists of a relative large amound of CPUs, which makes it the proper way for a large amound of pixel multiplications and matirx mathematical manipulations.
* The basic (and simplified) sobel edge detectors, which were implemented in thie projects, are having the same process:
    * Upload an image
    * Convert the image to a "greyscale image"
    * Define a kernel / Operator - A kernel is a matrix which chosen base on it's property to find the edges of an image. Basic kernels, which differ by their matrix values, are called "Sobel Operator", and "Laplacian Kernel".
    * Calculate the sum of products (convolution) between the greyscale image and the Kernel.

## Tools and Frameworks
* The project consists of 3 parts, which demonstrated the efficency according to the language which implements the alghorithem:
    *  Python - The fact Python is a high level language (written in C), makes it the most unefficient implementaion for the alghorithem. On the other hand, this is the simplest and easiest wat for implementation.
    * C++ - Using C++, which is low level language, significantly improves the results of the same basic alghorithem.
    * Cuda - Unline C++, which uses with "CPU", Cuda is additional language which used for Nvidia GPU's interactions, combined with C++. As a result, the Cuda implementaion reflects both CPU (using C++) and GPU (using Cuda) usage and results the best results. Because Cuda can be running only on Nvidia GPU, this part of code developed on the "Jetson Nano (2G Ram)" board.


## Installations
* Clone the repository.
* Create a folder for your project named "Edge_Detection" and move the content of the "Edge_Detection" repository directory, to the one you created.
* Python Instructions (Windows):
    * U
* Create an enviroment: `conda create name –-Edge_Detection python=3.6.13`
* activate the enviroment: `conda activate Edge_Detection`
* install the following libraries: 
`conda install pip install opencv-python==4.6.0`
`conda install Matplotlib=2.0.2`
`conda install pillow=5.1.0`


## Demonstration
TBD

## Notes
TBD 

## Repository-status
* Status - Beta
* TODOs - TBD