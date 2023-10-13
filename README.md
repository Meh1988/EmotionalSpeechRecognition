# EmotionalSpeechRecognition
Emotional Speech Recognition using RAVDESS
This repository contains code to classify emotional states from audio files using various machine learning and deep learning models. The dataset used is RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

Contents
Overview
Dataset
Installation
Usage
Model Architectures
Results
Acknowledgements
Overview
The primary goal is to analyze audio files and predict the emotion behind the spoken content. We process the audio files, extract features, and then apply machine learning and deep learning techniques to achieve this.

Dataset
RAVDESS is a validated multimodal database of emotional speech and song. For this project, we specifically focus on the speech part.

Each file in RAVDESS is named in such a way that an experienced user can define the emotion, speech content, and more just from the name.

Installation
Clone the repository:
bash


Install required packages:
pip install librosa tqdm keras scikit-learn matplotlib
pip install librosa tqdm keras scikit-learn matplotlib
pip install librosa
pip install librosa scikit-learn keras matplotlib

Ensure you have the RAVDESS dataset downloaded on your machine.

Usage
Adjust the dataset_path variable in the script to point to the location of your RAVDESS dataset.

Run the script:
python EmotionalSpeech.py

Model Architectures
Random Forest: A traditional machine learning model.
SVM (Support Vector Machine): Another traditional machine learning model.
Simple Neural Network: A basic feed-forward neural network.
CNN (Convolutional Neural Network): A deep learning model, generally used for image classification but in our case, it's applied to MFCCs extracted from audio files.
Results
The models' performances are evaluated using accuracy as a metric. The results are displayed as a bar chart for easy comparison between the models.

Acknowledgements
Thanks to the creators of RAVDESS dataset for providing an extensive dataset for emotional analysis:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/

