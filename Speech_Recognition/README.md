# Speech Recognition
## Table of contents
* [Overview](#Overview)
* [Tools and Frameworks](#Tools-and-Frameworks)
* [Resources](#Resources)
* [Installations](#Installations)
* [Demonstration](#Demonstration)
* [Notes](#Notes)
* [Repository status](#Repository-status)

## Overview
* The purpose of this repository is to demonstrate speech recognition base on:
    * Audio File
    * Microphone 
* It consists of the following parts:
    * Part 1 - AudioFile.py and AudioFileNoise.py:  Demonstrated a given audio file which is transcripted into text. There are 2 audio files for part 1. One which is without any ambient noise and another with noise. These two python scripts related to those files and demonstrate how to make a speech recognition based on the files with and without noise.
    * Part 2 - MicrophoneNoise: Demonstrated how one can use it's microphone in an noisy environment and get the transcription of his live recorder voise.
* In this repository I rely on "speech_recognition" module and assign the "recognize_google()" Recognizer. It's worth mentioning there are additional recognizer, as described bellow, but the reason I use google's is beacuse it has hard coded API key which make it much easier to use without the need establishing an authentication service. As said above, additional recognizers are:
    * recognize_bing(): Microsoft Bing Speech
    * recognize_google(): Google Web Speech API
    * recognize_google_cloud(): Google Cloud Speech - requires installation of the google-cloud-speech package
    * recognize_houndify(): Houndify by SoundHound
    * recognize_ibm(): IBM Speech to Text
    * recognize_sphinx(): CMU Sphinx - requires installing PocketSphinx
    * recognize_wit(): Wit.ai

## Resources
* Real Python - [Tutorial - The Ultimate Guide To Speech Recognition With Python](https://realpython.com/python-speech-recognition/#working-with-microphones). The repository relies on this tutorial.
* Real Python - [Course - Learn Text Classification With Python and Keras](https://realpython.com/lessons/speech-recognition-python-overview/)
* Stackoverflow - [Supported languiages by recognize_google() service](https://stackoverflow.com/questions/14257598/what-are-language-codes-in-chromes-implementation-of-the-html5-speech-recogniti)

## Installations
Please follow the installation instruction in the Readme file of the reposiroty (Machine_Learning/Readme) before the instructions in this section.

* Create a folder for your project
* Open Visual code in the project destination
* Create an enviroment: `conda create name –-Speech_Recognition python=3.8`
* activate the enviroment: `conda activate Speech_Recognition`
* install the following libraries: 
`conda install SpeechRecognition`
`conda install pyaudio` (requires specific python version such as python == 3.8)

## Demonstration
NTA

## Notes
NTA

## Repository-status
* Status - Beta
