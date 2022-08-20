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
* The fact there are a relative significant amount of pixels to multiply and summarize, due to the charctericties of this algorithm, this calculation is complex for the CPU alone, which workd in a procedural method. GPU, unlike CPU, consists of a relative large amount of CPUs, which makes it the proper way for a large amound of pixel multiplications and matirx mathematical manipulations.
* The basic (and simplified) sobel edge detectors, which were implemented in these projects, are having the same process:
    * Upload an image
    * Convert the image to a "greyscale image"
    * Define a kernel / Operator - A kernel is a matrix which chosen base on it's property to find the edges of an image. Common edge detection kernels, which differ by their matrix values, are called "Sobel Operator", and "Laplacian Kernel".
    * Calculate the sum of products (convolution) between the greyscale image and the Kernel. This operation results the "edges" image


## Tools and Frameworks
* The project consists of 3 parts, which demonstrated the efficency according to the language which implements the algorithm:
    *  Python - The fact Python is a high level language (written in C), makes it the most unefficient implementaion for the alghorithem. On the other hand, this is the simplest and easiest way for implementation.
    * C++ - Using C++, which is a low level language, significantly improves the results of the same basic algorithm.
    * Cuda - Unline C++ commands, which activates "CPU", Cuda is an additional language which used for Nvidia GPU's interactions, combined with C++. As a result, the Cuda implementaion reflects both to CPU (using C++) and GPU (using Cuda) usage and results the best results. Because Cuda can be running only on Nvidia GPU, this part of code developed on the *"Jetson Nano"* (2G Ram) board.


## Installations and Usage
* Clone the repository using `git clone <repo link>` command
* Create a folder for your project named "Edge_Detection" and move the content of the "Edge_Detection" repository directory, to the one you created.
* Instructions for running *Python* edge detector (Windows):
    * Create an enviroment: Conda environment creation from a given yml can be created using `conda env create --file .\env_windows.yml` command. You can create your own environment manually by using `conda create name –-Edge_Detection_Py python=3.6.13` (Unrecomended, due the need to manually install all relevant packages and face package incompatability)
    * activate the enviroment: `conda activate Edge_Detection_Py`
    * Run the script: `Python Edge_Detection_Py`
* Instructions for running *Python* edge detector (Linux):
    * Repeat the steps, but use use the env_linux.yml file (instead env_windows.yml)
* Instructions for running *CPP* edge detector (Windows):
    * Install Visual Studio (The instruction bellow reffer to VS IDE)
    * Download [CV library](https://sourceforge.net/projects/opencvlibrary/)
    * Config your VS IDE according to this [tutorial](https://learnopencv.com/code-opencv-in-visual-studio/). In case you are running into linkage issues, use this turorial[https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html].
    * Make sure your system environmental varables includes the path to the bin directory of the OpenCV library which was previously downloaded.
    * Compile the solution from the IDE
* Instructions for running *CPP* edge detector (Linux):
    * Please refer to the "TODOs"
* Instructions for running *Cuda* edge detector (Linux) - As previously stated, this code must run on the Jetson Nano, because it include an [Nvidia GPU](https://developer.nvidia.com/embedded/jetson-nano-2gb-developer-kit).
    * Make sure Cuda installed on your board using this [tutorial](https://maker.pro/nvidia-jetson/tutorial/introduction-to-cuda-programming-with-jetson-nano)
    * Compile the main.cu using the './compile.sh' command
    * Run the project using the `./run_experiments.sh imgs_in/image_original.jpg` command


## Demonstration
Python results:

* Windows OS (Architecture: I7, 64 bit) - 4 second approximatly:

<kbd> <p align="center">
  <img style="display: block;margin-left: auto;margin-right: auto; width: 50%; height: 50%;" src="https://github.com/gy-m/Machine_Learning/blob/master/Edge_Detection/Py/Windows/Demonstrations/Demonstration.jpg?raw=true">
</p> </kbd>

* Linux OS (Architecture: Quad-core ARM® A57 @ 1.43 GHz) - **35 second approximatly**:

<kbd> <p align="center">
  <img style="display: block;margin-left: auto;margin-right: auto; width: 50%; height: 50%;" src="https://github.com/gy-m/Machine_Learning/blob/master/Edge_Detection/Py/Linux/Demonstrations/Demonstration.jpg?raw=true">
</p> </kbd>

C++ results:

* Windows OS (Architecture: I7, 64 bit) - **1 second approximatly**:

<kbd> <p align="center">
  <img style="display: block;margin-left: auto;margin-right: auto; width: 50%; height: 50%;" src="https://github.com/gy-m/Machine_Learning/blob/master/Edge_Detection/CPP/Windows/Demonstrations/Demonstration.jpg?raw=true">
</p> </kbd>

Cuda rsults:

* Linux OS (Architecture: Quad-core ARM® A57 @ 1.43 GHz, using the [GPU](https://developer.nvidia.com/embedded/jetson-nano-2gb-developer-kit)) - **Average of 240 miliseconds approximatly** for evry GPU iteration:

<kbd> <p align="center">
  <img style="display: block;margin-left: auto;margin-right: auto; width: 50%; height: 50%;" src="https://github.com/gy-m/Machine_Learning/blob/master/Edge_Detection/Cuda/Demonstrations/Demonstration.jpg?raw=true">
</p> </kbd>


## Notes
* Conda environment was created using  `conda export env > <file name.yml>` therefor suggested to create the conda environment using `conda env create --file <file name.yml>` command


## Repository Status
* Status - Beta
* TODOs:
    * CPP project - Adding time function for convolution calculation time evaluation
    * Py project - Calculate the convolution time instead of programming running time
    * CPP/Linux project - Configure OpenCV on the jetson nano and demonstrate the results (in addition to the CPP windows demonstrations), and update "Installations" and "Demonstration" sections of this file accordingly.