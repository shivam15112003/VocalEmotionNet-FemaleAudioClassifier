##🧠 Methodology: Fine-Tuned Female English Voice Emotion Detection System

This document outlines the complete pipeline for developing an AI-based emotion detection system tailored to female English voice input, using deep learning, audio processing, fine-tuned VGG16, real-time gender & language verification, and a Tkinter GUI.

---

##1️⃣ Data Collection & Organization

• Audio samples categorized into 7 emotions:
  - angry, disgust, fear, happy, neutral, pleasant, sad
• Stored in labeled subdirectories inside the datasets/ folder.
• Each audio clip represents one dominant emotion.

---

##2️⃣ Feature Extraction (Audio to Image Conversion)

• Loaded audio files using librosa.load().
• Computed Mel spectrograms to represent time-frequency patterns.
• Resized spectrograms to 224x224 pixels to fit VGG16 input requirements using TensorFlow.
• Converted single-channel spectrograms into 3-channel RGB images for compatibility with pre-trained VGG16.

---

##3️⃣ Model Architecture & Transfer Learning with Fine-Tuning

• Used VGG16 pre-trained on ImageNet as base model.
• Initially froze all VGG16 layers.
• Then UNFROZE the top 20 layers of VGG16 to adapt to Mel spectrogram patterns.
• Custom classifier head architecture:
  - Flatten layer
  - Dense(512, ReLU)
  - Dropout(0.5)
  - Dense(7, softmax) for multi-class emotion output.
• Compiled using Adam optimizer with low learning rate (1e-5) and categorical crossentropy loss.

---

##4️⃣ Training & Evaluation

• Encoded emotion labels using LabelEncoder and one-hot encoded using to_categorical.
• Split dataset into 80% training and 20% testing using train_test_split.
• Trained for 15 epochs with validation data.
• Evaluated using accuracy score from scikit-learn after fine-tuning.

---

##5️⃣ Real-Time Inference Pipeline

• Built an interactive GUI using Tkinter.
• Two input modes supported:
  - File upload of audio (.wav/.mp3)
  - Real-time microphone recording (5 seconds)
• On inference:
  1. Extract Mel spectrogram from input audio.
  2. Estimate average pitch using librosa.piptrack().
  3. Apply gender filter: accept only female pitch range (~165–250 Hz).
  4. Detect spoken language using SpeechBrain pretrained model.
  5. Apply language filter: accept only English (en).
  6. Pass processed spectrogram to fine-tuned VGG16 model for emotion classification.
  7. Display predicted emotion in GUI.

---

##6️⃣ Constraints & Assumptions

• The system strictly accepts female voices speaking English.
• Filters out male, non-English, or very high-pitched voices to maintain accuracy.
• Relies on Mel spectrogram image representations for audio classification instead of raw waveform data.

---

##✅ Outcome

• Fully integrated real-time fine-tuned emotion detection system.
• Achieved improved test accuracy (~95%) after fine-tuning VGG16.
• Incorporates robust gender & language verification before classification.
• Provides an end-to-end interactive user interface for practical deployment.