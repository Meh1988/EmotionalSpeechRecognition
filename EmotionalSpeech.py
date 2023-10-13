import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# 1. Load & Preprocess RAVDESS Dataset
dataset_path = "RAVDESS"  # Ensure this points to your RAVDESS dataset location

data = []
labels = []

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

max_pad_len = 400  # Define a maximum padding length

def pad_mfcc(mfcc):
    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]  # Truncate to max_pad_len
    else:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

for actor_dir in tqdm(os.listdir(dataset_path), desc="Processing audio files"):
    actor_path = os.path.join(dataset_path, actor_dir)
    for audio_file in os.listdir(actor_path):
        if len(audio_file.split("-")) < 3:
            print(f"Unexpected file format: {audio_file}. Skipping this file.")
            continue

        emotion_key = audio_file.split("-")[2]
        emotion = emotions[emotion_key]
        
        full_path = os.path.join(actor_path, audio_file)
        y, sr = librosa.load(full_path, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        mfcc = pad_mfcc(mfcc)  # Pad the MFCC to the defined length
        
        data.append(mfcc)
        labels.append(list(emotions.values()).index(emotion))

data = np.array(data)
labels = to_categorical(np.array(labels))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 2. Develop Classifiers

# Random Forest
print("\nTraining Random Forest...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

rf = RandomForestClassifier()
rf.fit(X_train_flat, y_train.argmax(axis=1))
rf_preds = rf.predict(X_test_flat)
rf_acc = accuracy_score(y_test.argmax(axis=1), rf_preds)

# SVM
print("\nTraining SVM...")
svm = SVC()
svm.fit(X_train_flat, y_train.argmax(axis=1))
svm_preds = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test.argmax(axis=1), svm_preds)

# Simple Neural Network
print("\nTraining Simple Neural Network...")
nn_model = Sequential()
nn_model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
nn_model.add(Dense(512, activation='relu'))
nn_model.add(Dense(256, activation='relu'))
nn_model.add(Dense(len(emotions), activation='softmax'))

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
nn_acc = nn_model.evaluate(X_test, y_test)[1]

# CNN
print("\nTraining CNN...")
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(len(emotions), activation='softmax'))

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, validation_split=0.1)
cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)[1]

# 3. Visualize Results
models = ['Random Forest', 'SVM', 'Simple NN', 'CNN']
accuracies = [rf_acc, svm_acc, nn_acc, cnn_acc]

plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Emotion Classification Accuracy Comparison')
plt.show()
