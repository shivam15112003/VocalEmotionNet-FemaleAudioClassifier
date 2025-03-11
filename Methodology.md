# 📌 Methodology

## 1️⃣ Data Collection & Preprocessing
- Collected **female voice recordings** from multiple sources.
- Applied **Mel-Frequency Cepstral Coefficients (MFCC)** and **Mel Spectrogram** extraction using **Librosa**.
- Normalized and augmented audio data for improved generalization.

## 2️⃣ Model Architecture & Training
- Used **VGG16 pre-trained model** as the feature extractor.
- Added custom **Dense and Dropout layers** for classification.
- Compiled with **Adam optimizer** and **categorical cross-entropy loss**.
- Trained with labeled datasets, achieving **94% accuracy**.

## 3️⃣ Emotion Detection & Classification
- Used **softmax activation** for multi-class emotion classification.
- Predicted emotions such as **happy, sad, angry, neutral**, etc.
- Evaluated model performance using **training and test datasets**.

## 4️⃣ GUI & Real-Time Detection
- Integrated **Tkinter** for a user-friendly **audio recording & upload** interface.
- Enabled **live voice recording** and emotion prediction.
- Displayed real-time emotion detection results using **message pop-ups**.

This methodology ensures **efficient and accurate real-time emotion recognition** from **female voice recordings** for applications in **mental health analysis, AI assistants, and human-computer interaction**.
