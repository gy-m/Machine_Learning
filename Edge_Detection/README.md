# Edge Detection
## Table of contents
* [Overview](#Overview)
* [Tools and Frameworks](#Tools-and-Frameworks)
* [Installations and Usage](#Installations-and-Usage)
* [Demonstration](#Demonstration)
* [Notes](#Notes)
* [Repository Status](#Repository-status)


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


## Installations and Usage
* Clone the repository.
* Create a folder for your project named "Edge_Detection" and move the content of the "Edge_Detection/Py" repository directory, to the one you created.
* Python Instructions (Windows):
    * Create an enviroment: Conda environment creation from a given yml can be created using `conda env create --file .\env_windows.yml` command. You can create your own environment manually by using `conda create name –-Edge_Detection_Py python=3.6.13` (Unrecomended, due the need to manually install all relevant packages and face package incompatability)
    * activate the enviroment: `conda activate Edge_Detection_Py`
    * Run the script: `Python Edge_Detection_Py`
* Python Instructions (Linux):
    * Repeat the steps, but use use the env_linux.yml file (instead env_windows.yml)
* CPP Instructions (Windows):
    * Install Visual Studio (The instruction bellow reffer to VS IDE)
    * Download [CV library](https://sourceforge.net/projects/opencvlibrary/)
    * Config your VS IDE according to this [tutorial](https://learnopencv.com/code-opencv-in-visual-studio/). In case you are running into linkage issues, use this turorial[https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html].
    * Make sure your system environmental varables includes the path to the bin directory of the OpenCV library which was previously downloaded.
    * Compile the solution from the IDE
* CPP Instructions (Linux):
    * Please refer to the "TODOs"
* Cuda Instructions - As previously stated, this code must run on the Jetson Nano, because it include an Nvidia GPU.
    * Make sure Cuda installed on your board using this [tutorial](https://maker.pro/nvidia-jetson/tutorial/introduction-to-cuda-programming-with-jetson-nano)
    * Compile the main.cu using the './compile.sh' command
    * Run the project using the `./run_experiments.sh imgs_in/image_original.jpg` command


## Demonstration
* Python results:
    * Windows (I7, 64 bit):

<kbd> <p align="center">
  <img style="display: block;margin-left: auto;margin-right: auto; width: 50%; height: 50%;" src="https://github.com/gy-m/Machine_Learning/blob/master/Edge_Detection/Py/Windows/Demonstrations/Demonstraion.jpg?raw=true">
</p> </kbd>


## Notes
TBD 


## Repository Status
* Status - Beta
* TODOs - Fix CPP/Linux project, and update "Installations" section of this file accordingly.