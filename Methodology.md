# 🧠 Methodology: Female Voice Emotion Detection System

This document outlines the complete pipeline for developing an AI-based emotion detection system tailored to **female English voice input**, using **deep learning**, **audio processing**, and a real-time **Tkinter GUI**.

---

## 1️⃣ Data Collection & Organization

* Audio samples categorized by emotion:

  * angry, disgust, fear, happy, neutral, pleasant, sad
* Stored in labeled directories within a `datasets/` folder
* Each audio clip assumed to represent one dominant emotion

---

## 2️⃣ Feature Extraction (Audio to Image Conversion)

* Loaded audio using `librosa.load()`
* Generated **Mel spectrograms** to represent time-frequency patterns
* Resized all spectrograms to **224×224** pixels using TensorFlow for VGG16 input
* Converted spectrograms to **3-channel images** by repeating the array across RGB channels

---

## 3️⃣ Model Architecture & Transfer Learning

* Used **VGG16** pretrained on ImageNet as a feature extractor (frozen base)
* Custom classifier head:

  * Flatten layer
  * Dense(512, relu)
  * Dropout(0.5)
  * Dense(7, softmax)
* Compiled with Adam optimizer and categorical crossentropy loss

---

## 4️⃣ Training & Evaluation

* Encoded emotion labels with `LabelEncoder` + `to_categorical`
* Split dataset 80:20 for training and testing
* Trained for 5 epochs with validation
* Evaluated using **accuracy score** from scikit-learn

---

## 5️⃣ Real-Time Inference Pipeline

* Built a GUI with Tkinter
* Supported two input modes:

  * Audio file upload
  * Microphone-based voice recording (5 seconds)
* On submission:

  1. Load and preprocess audio
  2. Extract spectrogram, resize, convert to RGB format
  3. Estimate **average pitch** using `librosa.piptrack()`
  4. Filter: accept only **female pitch range (180–225 Hz)**
  5. Run prediction and display emotion in GUI

---

## 6️⃣ Constraints & Assumptions

* System only works with **female voices speaking English**
* Ignores audio with low or male pitch range to improve emotion classification accuracy
* Training and inference performed on Mel spectrograms instead of raw waveform

---

## ✅ Outcome

* Real-time audio-based emotion detector with \~94% test accuracy
* Efficient and interactive GUI
* Reusable pipeline for emotion classification from speech
